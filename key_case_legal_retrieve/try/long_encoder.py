# coding: utf-8
'''
长文本编码器
'''

from tqdm import tqdm
import json
import os, re, jieba
import numpy as np
import torch
import random
import argparse
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from gensim.summarization import bm25
from transformers import AutoModel, AutoTokenizer


data_type = 'test'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='cuda', type=str)
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-label_file', default='./data/phase_2/train/label_top30_dict.json', type=str)
    parser.add_argument('-dev_label_file', default='./data/phase_1/train/label_top30_dict.json', type=str)
    parser.add_argument('-query_file', default='./data/phase_2/test/query.json', type=str)
    parser.add_argument('-candidate_file', default='./data/phase_2/test/candidates/', type=str)
    parser.add_argument('-saved_embeddings', default='./data/phase_2/train_embeddings_s', type=str)
    parser.add_argument('-saved_dev_embeddings', default='./data/phase_2/dev_embeddings_s', type=str)
    parser.add_argument('-saved_test_embeddings', default='./data/phase_2/test_embeddings_s', type=str)
    parser.add_argument('-saved_data', default='./data/phase_2/train_data_win200.json', type=str)
    parser.add_argument('-saved_dev_data', default='./data/phase_2/dev_data_win200.json', type=str)
    parser.add_argument('-saved_test_data', default='./data/phase_2/test_data_win200.json', type=str)
    parser.add_argument('-model_path', default='../pretrained_model/lawformer', type=str)
    parser.add_argument('-output_path', default='./result', type=str)
    parser.add_argument('-max_length', default=1533, type=int)
    parser.add_argument('-batch_size', default=10, type=int)
    args = parser.parse_args()

    # config environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("../pretrained_model/roberta")
    tokenizer.add_tokens(['☢'])
    model = LongModel(args.model_path, tokenizer).cuda()
    model.load_state_dict(torch.load('result/bsz-1-lr-1e-05/2epoch_0.9386_bert.pt'))
    # BatchNorm = BatchNormMLP().cuda()
    model.eval()
    longdataset = LongDataset(args.saved_data, args.query_file, args.candidate_file, args.label_file,
                              args.dev_label_file, args.saved_dev_data, args.saved_test_data, tokenizer)
    dataset = DataLoader(longdataset, batch_size=args.batch_size, shuffle=False)
    all_embeddings, batch_embeddings, i = [], None, 0
    for data in tqdm(dataset, desc='数据向量化'):
        torch.cuda.empty_cache()
        i += 1
        for k, v in data.items():
            data[k] = v.view(2*args.batch_size, -1).cuda()
        embeddings = model(**data)
        if i == 1:
            batch_embeddings = embeddings
            continue
        batch_embeddings = torch.cat([batch_embeddings, embeddings], 0)
        if i == 100 // args.batch_size:
            # TODO Batch Normalization
            # batch_embeddings = BatchNorm(batch_embeddings)
            # embeddings = torch.mean(batch_embeddings, dim=1)
            all_embeddings.append(batch_embeddings.cpu().detach().numpy())
            i = 0

    print('The number of {} data: {}'.format(data_type, len(all_embeddings)))
    if data_type == 'train':
        np.save(args.saved_embeddings, all_embeddings)
        print(u'输出路径：%s' % args.saved_embeddings)
    elif data_type == 'test':
        np.save(args.saved_test_embeddings, all_embeddings)
        print(u'输出路径：%s' % args.saved_test_embeddings)
    else:
        np.save(args.saved_dev_embeddings, all_embeddings)
        print(u'输出路径：%s' % args.saved_dev_embeddings)


def seed_everything(seed):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


class BatchNormMLP(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        in_dim = hidden_size
        hidden_dim = hidden_size * 2
        out_dim = hidden_size
        affine = False
        list_layers = [nn.Linear(in_dim, hidden_dim, bias=False),
                       nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(hidden_dim, out_dim, bias=False),
                        nn.BatchNorm1d(out_dim, affine=affine)]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x)


class LongModel(nn.Module):
    def __init__(self, model_path, tokenizer):
        super(LongModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.model.resize_token_embeddings(len(tokenizer))
        self.linear = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            output = self.model(input_ids, attention_mask, token_type_ids)
            # pre_embedding = torch.mean(output[0][1::2, 1:510], dim=1)
            # post_embedding = torch.mean(output[0][1::2, 511:-1], dim=1)
            case_cls = output[1][1::2]
            key_cls = output[1][::2]
            key_logits = self.linear(key_cls)
            case_logits = self.linear(case_cls)
        return torch.cat([key_cls, case_cls, key_logits, case_logits], -1)


class LongDataset(Dataset):
    def __init__(self, data_file, query_file, candidate_dir, label_top30_file, dev_label_file, saved_dev_file, saved_test_file, tokenizer):
        self.tokenizer = tokenizer
        if dev_label_file:
            self.filter_qids = json.load(open(dev_label_file, 'r', encoding='utf8')).keys()
            # TODO: Add Hard Samples
        if os.path.exists(data_file):
            if data_type == 'train':
                with open(data_file, 'r', encoding='utf8') as f:
                    self.data = json.load(f)
                    print(len(self.data))
            elif data_type == 'test':
                with open(saved_test_file, 'r', encoding='utf8') as f:
                    self.data = json.load(f)
            else:
                with open(saved_dev_file, 'r', encoding='utf8') as f:
                    self.data = json.load(f)
        else:
            if data_type == 'test':
                self.data = self.load_data(query_file, candidate_dir, data_file, None, saved_dev_file, saved_test_file)
            else:
                self.data = self.load_data(query_file, candidate_dir, data_file, label_top30_file, saved_dev_file, saved_test_file)
                exit()

    def load_data(self, query_file, candidate_dir, saved_file, label_top30_file, saved_dev_file, saved_test_file):
        queries = []
        fq = open(query_file, 'r', encoding='utf8')
        for line in fq:
            queries.append(json.loads(line.strip()))
        all_label_top30 = json.load(open(label_top30_file, 'r', encoding='utf8')) if label_top30_file else None
        data, dev_data, test_data = [], [], []
        for query in tqdm(queries, desc=u'数据转换'):
            qidx, q, crime = str(query['ridx']), str(query['q']), '、'.join(query['crime'])
            # if qidx in self.filter_qids:
            #     continue
            doc_dir = os.path.join(candidate_dir, qidx)
            doc_files = os.listdir(doc_dir)
            if len(doc_files) != 100:
                print(qidx)
                exit(0)
            for doc_file in doc_files:
                doc_path = os.path.join(doc_dir, doc_file)
                didx = str(doc_file.split('.')[0])
                with open(doc_path, 'r', encoding='utf8') as fd:
                    sample_d = json.load(fd)
                ajjbqk, ajName = sample_d['ajjbqk'], sample_d['ajName']
                # try:
                #     cpfxgc = sample_d['cpfxgc']
                #     cpfxgc = cpfxgc_filter(cpfxgc)[:100]
                #     if len(cpfxgc) < 15:
                #         cpfxgc = sample_d['ajName'] + '，' + cpfxgc
                # except:
                #     cpfxgc = sample_d['ajName']
                ajjbqk = filter_jbqk(q, ajjbqk).strip()
                if label_top30_file:
                    label = all_label_top30[qidx][didx] if didx in all_label_top30[qidx] else 0
                    all_label = [qidx, didx, label]
                    if qidx in self.filter_qids:
                        dev_data.append({'crime': crime, 'query': q, 'ajName': ajName, 'candidate': ajjbqk, 'labels': all_label})
                    else:
                        data.append({'crime': crime, 'query': q, 'ajName': ajName, 'candidate': ajjbqk, 'labels': all_label})
                else:
                    all_label = [qidx, didx]
                    test_data.append({'crime': crime, 'query': q, 'ajName': ajName, 'candidate': ajjbqk, 'labels': all_label})
        if data_type == 'train':
            print(len(data))
            with open(saved_file, 'w', encoding='utf8') as fs:
                json.dump(data, fs, ensure_ascii=False, indent=2)
            # return data
        # if data_type == 'dev':
            print(len(dev_data))
            with open(saved_dev_file, 'w', encoding='utf8') as fd:
                json.dump(dev_data, fd, ensure_ascii=False, indent=2)
            return dev_data
        else:
            print(len(test_data))
            with open(saved_test_file, 'w', encoding='utf8') as ft:
                json.dump(test_data, ft, ensure_ascii=False, indent=2)
            return test_data

    def __getitem__(self, index):
        data = self.data[index]

        q_crime_tokens, d_crime_tokens = self.tokenizer.tokenize(data['crime']), self.tokenizer.tokenize(data['ajName'])
        crime_tokens = ['[CLS]'] + q_crime_tokens + ['[SEP]'] + d_crime_tokens + ['[SEP]']
        crime_ids = self.tokenizer.convert_tokens_to_ids(crime_tokens)
        crime_types = [0] * (len(q_crime_tokens) + 2) + [1] * (len(d_crime_tokens) + 1)

        query_cut = self.tokenizer.tokenize(data['crime']+'☢'+data['query'])[:509]
        candidate_cut = self.tokenizer.tokenize(data['ajName']+'☢'+data['candidate'])[:1020]
        tokens = ['[CLS]'] + query_cut + ['[SEP]'] + candidate_cut + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        types = [0] * (len(query_cut) + 2) + [1] * (len(candidate_cut) + 1)

        input_ids, token_type_ids, attention_mask = self.pad_seq([crime_ids, token_ids], [crime_types, types])

        feature = {'input_ids': torch.LongTensor(input_ids),
                   'token_type_ids': torch.LongTensor(token_type_ids),
                   'attention_mask': torch.LongTensor(attention_mask),
                   }
        return feature

    def pad_seq(self, ids_list, types_list):
        batch_len = 1533
        new_ids_list, new_types_list, new_masks_list = [], [], []
        for ids, types in zip(ids_list, types_list):
            masks = [1] * len(ids) + [0] * (batch_len - len(ids))
            types += [0] * (batch_len - len(ids))
            ids += [0] * (batch_len - len(ids))
            new_ids_list.append(ids)
            new_types_list.append(types)
            new_masks_list.append(masks)
        return new_ids_list, new_types_list, new_masks_list

    def __len__(self):
        return len(self.data)


def cpfxgc_filter(cpfxgc):
    cpfxgc_first_list = cpfxgc.split('。')
    cpfxgc_list = [s for s in re.split(r'[。；]', '。'.join(cpfxgc_first_list[1:])) if len(s) > 0]
    filter_cpfxgc = []
    for sent in cpfxgc_list:
        key_word = [['行为', '罪'], ['罪', '事实清楚']]
        or_word = ['已构成', '已经构成', '符合']
        if (all(word in sent for word in key_word[0]) or all(word in sent for word in key_word[1])) \
                and any(word in sent for word in or_word):
            filter_cpfxgc.append(sent)
    if filter_cpfxgc != []:
        filter_cpfxgc.insert(0, cpfxgc_first_list[0])
        filter_cpfxgc = '。'.join(filter_cpfxgc)
    else:
        filter_cpfxgc = cpfxgc_first_list[0]
    cpfxgc = string_rule(filter_cpfxgc)[1:]
    return cpfxgc


def string_rule(string):
    p = re.compile(r'[（](.*?)[）]', re.S)
    b = re.compile(r'(被告人)(.*?)(等人)', re.S)
    string = re.sub(b, '', string)
    string = re.sub(p, '', string)
    string = string.replace('被告人', '')
    string = string.replace('本院认为', '')
    return string


def filter_jbqk(query, doc, pre_sent_size=5, group_size=700, windows_size=200, select=1):
    sents_list = [s for s in re.split(r'[。！；？]', doc) if len(s) > 0]
    pre_sents = sents_list[:pre_sent_size]  # 前5句
    # # 用query中的每个句子去搜索更有效果
    group_sents = []
    sents_string = '。'.join(sents_list[pre_sent_size:])
    if len(sents_string) // group_size == 0:
        return doc
    for i in range(0, len(sents_string), windows_size):
        group_sents.append(sents_string[i:i + group_size])
    # 筛选出最相关的块（每块6句）
    rel_sents, scores = search_for_related_sents(query, group_sents, select=select)
    for s in [s for s in re.split(r'[。！；，：？]', rel_sents[0]) if len(s) > 0]:
        if s not in pre_sents:
            pre_sents.append(s)
    filter_sents = '。'.join(pre_sents)
    return filter_sents


def search_for_related_sents(query, sents_list, select=1):
    corpus = []
    with open('./stopword.txt', 'r', encoding='utf8') as g:
        words = g.readlines()
    stopwords = [i.strip() for i in words]
    stopwords.extend(['.','（','）','-','×'])
    for sent in sents_list:
        sent_tokens = [w for w in jieba.lcut(sent.strip()) if w not in stopwords]
        corpus.append(sent_tokens)
    bm25model = bm25.BM25(corpus)

    q_tokens = [w for w in jieba.lcut(query) if w not in stopwords]
    scores = bm25model.get_scores(q_tokens)
    rank_index = np.array(scores).argsort().tolist()[::-1]
    rank_index = rank_index[:select]
    return [sents_list[i] for i in rank_index], [scores[i] for i in rank_index]


if __name__ == '__main__':
    main()
