# coding: utf-8
import os, re
import json
import numpy as np
import networkx as nx
from tqdm import tqdm
import jieba
import matplotlib.pyplot as plt
from collections import OrderedDict


stop_words = set()
for w in open('./stopword.txt', encoding='utf8'):
    stop_words.add(w.strip())
stop_words_plus = list(stop_words)
for c in ['！', '。', '；', '，', '、', '：', '？']:
    stop_words_plus.remove(c)


class Doc(object):
    def __init__(self, text):
        self.text = text
        self.query = ''.join(w for w in jieba.lcut(text['q']) if w not in stop_words_plus)
        self.qid = str(text['ridx'])

    def filter_candidate(self, doc_dir, doc_file):
        self.cid = doc_file.split('.')[0]
        doc_path = os.path.join(doc_dir, doc_file)
        with open(doc_path, 'r', encoding='utf8') as fd:
            sample_d = json.load(fd)
        try:
            ajjbqk, cpfxgc = sample_d['ajjbqk'], sample_d['cpfxgc']
        except:
            return 1
        cpfxgc_first_list = cpfxgc.split('。')
        cpfxgc_list = [s for s in re.split(r'[。！；：？]', '。'.join(cpfxgc_first_list[1:])) if len(s) > 0]
        filter_cpfxgc = []
        for sent in cpfxgc_list:
            key_word = ['行为', '罪']
            or_word = ['已构成', '已经构成']
            if all(word in sent for word in key_word) and any(word in sent for word in or_word):
                filter_cpfxgc.append(sent)
        if filter_cpfxgc != []:
            filter_cpfxgc.insert(0, cpfxgc_first_list[0])
            filter_cpfxgc = '，'.join(filter_cpfxgc) + '。'
        else:
            filter_cpfxgc = cpfxgc_first_list[0] + '。'
        self.candidate = ''.join(w for w in jieba.lcut(ajjbqk) if w not in stop_words_plus)
        self.candidate = filter_cpfxgc + self.candidate
        return 0

    def check_same_id(self):
        return True if self.qid == self.cid else False

    def filter_bracket(self, string):
        p = re.compile(r'[（](.*?)[）]', re.S)
        new_string = re.sub(p, '', string)
        return new_string

    def parse_sentence(self):
        self.dset1 = OrderedDict()
        self.dset2 = OrderedDict()
        self.query = self.filter_bracket(self.query)
        self.candidate = self.filter_bracket(self.candidate)
        for idx, sent in enumerate(self.query.split('。')):
            if sent == '':
                continue
            sid = '{}_{:02d}'.format(self.qid, idx + 1)
            self.dset1[sid] = sent.strip()
        for idx, sent in enumerate(self.candidate.split('。')):
            if sent == '':
                continue
            sid = '{}_{:02d}'.format(self.cid, idx + 1)
            self.dset2[sid] = sent.strip()

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
            candidate = self.fw_dset2[sid]
            if len(candidate) >= 5:
                self.fs_dset2[sid] = candidate

    def calc_sentence_sim(self, s1, s2):
        return len(set(s1) & set(s2)) / (np.log(len(s1)) + np.log(len(s2)))

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
        node_size = [self.node_weight[v] * 5000 for v in self.graph]
        nx.draw(self.graph, node_color=node_color, node_size=node_size, with_labels=True)

    def show_each_graph(self):
        node_size_1 = [self.node_weight_1[v] * 5000 for v in self.graph1]
        nx.draw(self.graph1, node_size=node_size_1, with_labels=True)
        plt.show()
        node_size_2 = [self.node_weight_2[v] * 5000 for v in self.graph2]
        nx.draw(self.graph2, node_size=node_size_2, with_labels=True)
        plt.show()

    def important_sentence(self, topk=(3, 3)):
        imp_s1 = []
        imp_s2 = []
        for sid in self.node_weight:
            if self.get_docid(sid) == self.qid:
                imp_s1.append([sid, self.node_weight[sid]])
            elif self.get_docid(sid) == self.cid:
                imp_s2.append([sid, self.node_weight[sid]])

        imp_s1 = sorted(imp_s1, key=lambda x: x[1], reverse=True)
        imp_s2 = sorted(imp_s2, key=lambda x: x[1], reverse=True)
        imp_s1_sorted = sorted(imp_s1[:topk[0]], key=lambda x: int(x[0].split('_')[1]))
        imp_s2_sorted = sorted(imp_s2[:topk[1]], key=lambda x: int(x[0].split('_')[1]))
        return imp_s1_sorted, imp_s2_sorted, self.qid, self.cid

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
        results = self.important_sentence(topk)
        for k, v in dist_s1:
            self.node_weight[k] -= 10
        for k, v in dist_s2:
            self.node_weight[k] -= 10
        return results

    def selected_sentence_2(self, disk=3, topk=1, exclude_title=True):
        imp_s1, imp_s2, _, __ = self.important_sentence(topk)
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


def prepare_key_data(query_file, label_top30_file, candidate_dir, filepath):
    queries = []
    fq = open(query_file, 'r', encoding='utf8')
    for line in fq:
        queries.append(json.loads(line.strip()))
    all_label_top30 = json.load(open(label_top30_file, 'r', encoding='utf8')) if label_top30_file else None
    out = []
    for line in tqdm(queries):
        doc_dir = os.path.join(candidate_dir, str(line['ridx']))
        doc_files = os.listdir(doc_dir)
        for doc_file in doc_files:
            doc = Doc(line)
            flag = doc.filter_candidate(doc_dir, doc_file)
            if doc.check_same_id() or flag:
                continue
            doc.parse_sentence()
            doc.filter_word(stop_words)
            doc.filter_sentence()
            doc.build_pair_graph()
            s1, s2, qidx, didx = doc.important_sentence((3, 3))
            d1 = []
            d2 = []
            for s in s1:
                d1.append(doc.dset1[s[0]])
            for s in s2:
                d2.append(doc.dset2[s[0]])
            d1 = '。'.join(d1)
            d2 = '。'.join(d2)
            # sent_split = sent.split('，')
            # for fragment in sent_split:
            #     if all(word in fragment for word in ['罪', '行为', '构成']) or all(
            #             word in fragment for word in ['罪', '事实清楚']):
            #         sent_split.remove(fragment)
            # sent = '，'.join(sent_split)
            if label_top30_file is not None:
                score = all_label_top30[qidx][didx] if didx in all_label_top30[qidx] else 0
                if score == 3 or score == 2:
                    score -= 2
                label = [qidx, didx, int(score)]
            else:
                label = [qidx, didx]
            out.append({'query': d1, 'candidate': d2, 'label': label})
    with open(filepath, 'w', encoding='utf8') as fs:
        json.dump(out, fs, ensure_ascii=False, indent=2)


print("Create Train Set...")
query_file = './data/phase_1/train/query.json'
label_top30_file = './data/phase_1/train/label_top30_dict.json'
candidate_dir = './data/phase_1/train/candidates/'
saved_file = 'data/case_dev.json'
prepare_key_data(query_file, label_top30_file, candidate_dir, saved_file)

print("Create Validation Set...")
query_file = './data/phase_2/train/query.json'
label_top30_file = './data/phase_2/train/label_top30_dict.json'
candidate_dir = './data/phase_2/train/candidates/'
saved_file = 'data/case_train.json'
prepare_key_data(query_file, label_top30_file, candidate_dir, saved_file)


