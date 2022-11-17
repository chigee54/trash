# coding: utf-8
import os
import json
import numpy as np
import networkx as nx
# import sys
# sys.path.insert(0, './data/')
import nlp_utils
import jieba
import matplotlib.pyplot as plt

from collections import OrderedDict


train_data = './data/new_train.json'
dev_data = './data/new_dev.json'
test_data = './new_test.json'


stop_words = set()
for w in open('./stopwords-zh.txt', encoding='utf8'):
    stop_words.add(w.strip())


class Doc(object):
    def __init__(self, text):
        self.text = text
        self.query = text['query']
        self.candidate = text['candidate']
        self.qid = text['label'][0]
        self.cid = text['label'][1]
        self.id_label = text['label']

    def check_same_id(self):
        return True if self.qid == self.cid else False

    def parse_sentence(self, append_title_node=False):
        self.dset1 = OrderedDict()
        self.dset2 = OrderedDict()
        for idx, sent in enumerate(nlp_utils.split_chinese_sentence(self.query)):
            sid = '{}_{:02d}'.format(self.qid, idx+1)
            self.dset1[sid] = sent.strip()
        for idx, sent in enumerate(nlp_utils.split_chinese_sentence(self.candidate)):
            sid = '{}_{:02d}'.format(self.cid, idx+1)
            self.dset2[sid] = sent.strip()
        # if append_title_node:
        #     sid = '{}-{:02d}'.format(self.id1, 0)
        #     self.dset1[sid] = self.t1.strip()
        #     sid = '{}-{:02d}'.format(self.id2, 0)
        #     self.dset2[sid] = self.t2.strip()
            
    def filter_word(self, stop_words):
        self.fw_dset1 = OrderedDict()
        self.fw_dset2 = OrderedDict()
        for sid in self.dset1:
            self.fw_dset1[sid] = [w for w in jieba.lcut(self.dset1[sid]) if w not in stop_words]
        for sid in self.dset2:
            self.fw_dset2[sid] = [w for w in jieba.lcut(self.dset2[sid]) if w not in stop_words]

    def filter_sentence(self):
        self.fs_dset1 = OrderedDict()
        self.fs_dset2 = OrderedDict()
        for sid in self.fw_dset1:
            if len(self.fw_dset1[sid]) >= 5:
                self.fs_dset1[sid] = self.fw_dset1[sid]
        for sid in self.fw_dset2:
            if len(self.fw_dset2[sid]) >= 5:
                self.fs_dset2[sid] = self.fw_dset2[sid]
              
    def calc_sentence_sim(self, s1, s2):
        return len(set(s1)&set(s2))/(np.log(len(s1))+np.log(len(s2)))

    def get_docid(self, sid):
        return sid.split('_')[0]
    
    def build_each_graph(self):
        def build_graph(dset):
            graph = nx.Graph()
            for sid in dset:
                graph.add_node(sid)
            for sid_i in dset:
                for sid_j in dset:
                    if sid_i == sid_j:
                        continue
                    sim = self.calc_sentence_sim(dset[sid_i], dset[sid_j])
                    if sim > 0:
                        graph.add_edge(sid_i, sid_j, weight=sim)
            return graph
        self.graph1 = build_graph(self.fs_dset1)
        self.node_weight_1 = nx.pagerank(self.graph1)
        self.graph2 = build_graph(self.fs_dset2)
        self.node_weight_2 = nx.pagerank(self.graph2)
    
    def build_pair_graph(self):
        graph = nx.Graph()
        all_sent = list(self.fs_dset1.keys()) + list(self.fs_dset2.keys())

        def get_node(sid):
            docid = self.get_docid(sid)
            if docid == self.qid:
                return self.fs_dset1[sid]
            elif docid == self.cid:
                return self.fs_dset2[sid]
            else:
                raise ValueError()

        for sid in all_sent:
            docid = self.get_docid(sid)
            if docid == self.qid:
                graph.add_node(sid, color='red')
            elif docid == self.cid:
                graph.add_node(sid, color='blue')
            else:
                raise ValueError()

        for sid_i in all_sent:
            for sid_j in all_sent:
                if sid_i == sid_j:
                    continue
                sim = self.calc_sentence_sim(get_node(sid_i), get_node(sid_j))
                if sim > 0:
                    graph.add_edge(sid_i, sid_j, weight=sim)
        self.graph = graph
        self.node_weight = nx.pagerank(self.graph)
        
    def show_pair_graph(self):
        node_color = [self.graph.nodes[v]['color'] for v in self.graph]
        node_size = [self.node_weight[v]*5000 for v in self.graph]
        nx.draw(self.graph, node_color=node_color, node_size=node_size, with_labels=True)
        
    def show_each_graph(self):
        node_size_1 = [self.node_weight_1[v]*5000 for v in self.graph1]
        nx.draw(self.graph1, node_size=node_size_1, with_labels=True)
        plt.show()
        node_size_2 = [self.node_weight_2[v]*5000 for v in self.graph2]
        nx.draw(self.graph2, node_size=node_size_2, with_labels=True)
        plt.show()
        
    def important_sentence(self, topk=(10, 40), exclude_title=True):
        # if exclude_title:
        #     node_t1 = '{}-{:02d}'.format(self.id1, 0)
        #     node_t2 = '{}-{:02d}'.format(self.id2, 0)
        #     if node_t1 in self.node_weight:
        #         self.node_weight[node_t1] = 0.0
        #     if node_t2 in self.node_weight:
        #         self.node_weight[node_t2] = 0.0
        imp_s1 = []
        imp_s2 = []
        for sid in self.node_weight:
            if self.get_docid(sid) == self.qid:
                imp_s1.append([sid, self.node_weight[sid]])
            elif self.get_docid(sid) == self.cid:
                imp_s2.append([sid, self.node_weight[sid]])

        imp_s1 = sorted(imp_s1, key=lambda x: x[1], reverse=True)
        imp_s2 = sorted(imp_s2, key=lambda x: x[1], reverse=True)
        imp_s1_sorted = sorted(imp_s1[:topk[0]], key=lambda x: x[0])
        imp_s2_sorted = sorted(imp_s2[:topk[1]], key=lambda x: x[0])
        return imp_s1_sorted, imp_s2_sorted, self.id_label
    
    def distinct_sentence(self, disk=3, exclude_title=True):
        if exclude_title:
            node_t1 = '{}-{:02d}'.format(self.qid, 0)
            node_t2 = '{}-{:02d}'.format(self.cid, 0)
            if node_t1 in self.node_weight_1:
                self.node_weight_1[node_t1] = 0.0
            if node_t2 in self.node_weight_2:
                self.node_weight_2[node_t2] = 0.0
        dist_s1 = sorted(self.node_weight_1.items(), key=lambda x: x[1], reverse=True)
        dist_s2 = sorted(self.node_weight_2.items(), key=lambda x: x[1], reverse=True)
        dist_s1_sorted = sorted(dist_s1[:disk], key=lambda x: x[0])
        dist_s2_sorted = sorted(dist_s2[:disk], key=lambda x: x[0])

        return dist_s1_sorted, dist_s2_sorted
    
    def selected_sentence_1(self, disk=1, topk=3, exclude_title=True):
        dist_s1, dist_s2 = self.distinct_sentence(disk, exclude_title)
        for k, v in dist_s1:
            self.node_weight[k] += 10
        for k, v in dist_s2:
            self.node_weight[k] += 10
        results = self.important_sentence(topk, exclude_title)
        for k, v in dist_s1:
            self.node_weight[k] -= 10
        for k, v in dist_s2:
            self.node_weight[k] -= 10
        return results
    
    def selected_sentence_2(self, disk=3, topk=1, exclude_title=True):
        imp_s1, imp_s2, _ = self.important_sentence(topk, exclude_title)
        for k, v in imp_s1:
            self.node_weight_1[k] += 10
        for k, v in imp_s2:
            self.node_weight_2[k] += 10
        results = self.distinct_sentence(disk, exclude_title)
        for k, v in imp_s1:
            self.node_weight_1[k] -= 10
        for k, v in imp_s2:
            self.node_weight_2[k] -= 10
        return results


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


from tqdm import tqdm
def create_dataset(filepath, datapath, need_label=True, append_title_node=False, append_title=False, append_keyword=False):
    out = []
    i = 0
    for line in tqdm(json.load(open(datapath, 'r', encoding='utf8'))):
        # i += 1
        # if i <= 10086:
        #     continue
        doc = Doc(line)
        if doc.check_same_id():
            continue
        doc.parse_sentence(append_title_node=append_title_node)
        doc.filter_word(stop_words)
        doc.filter_sentence()
        doc.build_pair_graph()
        #doc.build_each_graph()
        #s1 = s2 = []
        s1, s2, label = doc.important_sentence((8, 40))
        #s1, s2 = doc.selected_sentence_1(disk=3, topk=5)
        #s1, s2 = doc.selected_sentence_2(disk=5, topk=3)
        #s1, s2 = list(doc.dset1.keys())[:7], list(doc.dset2.keys())[:7]
        #s1 = [[x, 1] for x in s1]
        #s2 = [[x, 1] for x in s2]
        d1 = []
        d2 = []
        # if append_title:
        #     d1.append(doc.t1 + ' ☢')
        #     d2.append(doc.t2 + ' ☢')
        # if append_keyword:
        #     d1.append(doc.k1 + ' ☄')
        #     d2.append(doc.k2 + ' ☄')
        for s in s1:
            d1.append(doc.dset1[s[0]])
        for s in s2:
            d2.append(doc.dset2[s[0]])
        #for s in s1:
        #    d1.append(' '.join(['的'] * len(''.join(doc.dset1[s[0]].split()))))
        #for s in s2:
        #    d2.append(' '.join(['的'] * len(''.join(doc.dset2[s[0]].split()))))
        # if len(d1) == 0 or len(d2) == 0:
        #     print('Error')
        #     break
        d1 = ' '.join(d1)
        d2 = ' '.join(d2)
        out.append({'query': d1, 'candidate': d2, 'label': label})
        # fs.write(json.dumps(out, ensure_ascii=False, cls=NpEncoder) + '\n')
    with open(filepath, 'w', encoding='utf8') as fs:
        json.dump(out, fs, ensure_ascii=False, indent=2)

tag = 'textrank_filter'
save_folder = './data/{}'.format(tag)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# print("Create Train Set...")
# create_dataset('./data/{}/train.json'.format(tag), train_data,
#                append_title_node=False, append_title=False, append_keyword=False)
# print("Create Validation Set...")
# create_dataset('./data/{}/dev.json'.format(tag), dev_data,
#                append_title_node=False, append_title=False, append_keyword=False)
print("Create Test Set...")
create_dataset('./data/{}/test.json'.format(tag), test_data,
               append_title_node=False, append_title=False, append_keyword=False)


