import argparse
import json
import os, math
from tqdm import tqdm
import random
from gensim.summarization import bm25
import jieba
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM

parser = argparse.ArgumentParser(description="Help info.")
parser.add_argument('--input', type=str, default='./data/phase_2/train/', help='input path of the dataset directory.')
parser.add_argument('--output', type=str, default='./result/', help='output path of the prediction file.')

# #If you need models from the server:
# huggingface = '/work/mayixiao/CAIL2021/root/big/huggingface'
# tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.huggingface, "thunlp/Lawformer"))
# model = AutoModelForMaskedLM.from_pretrained(os.path.join(args.huggingface, "thunlp/Lawformer"))

args = parser.parse_args()
input_path = args.input
input_query_path = os.path.join(input_path, 'query.json')
input_candidate_path = os.path.join(input_path, 'candidates')
output_path = args.output


def ndcg(ranks, K):
    dcg_value = 0.
    idcg_value = 0.

    sranks = sorted(ranks, reverse=True)

    for i in range(0,K):
        logi = math.log(i+2,2)
        dcg_value += ranks[i] / logi
        idcg_value += sranks[i] / logi
    if idcg_value == 0.0:
        idcg_value += 0.00000001
    return dcg_value/idcg_value


def cal_ndcg(all_preds, all_labels):
    ndcgs = []
    for qidx, pred_ids in all_preds.items():
        did2rel = all_labels[qidx]
        ranks = [did2rel[str(idx)] if str(idx) in did2rel else 0 for idx in pred_ids]
        ndcgs.append(ndcg(ranks, 30))
        # print(f'********** qidx: {qidx} **********')
        # print(f'top30 pred_ids: {pred_ids}')
        # print(f'ranks: {ranks}')
    # print(ndcgs)
    return sum(ndcgs) / len(ndcgs)

from collections import OrderedDict
if __name__ == "__main__":
    print('begin...')
    result = {}
    with open(os.path.join(os.path.dirname(__file__), 'stopword.txt'), 'r', encoding='utf8') as g:
        words = g.readlines()
    stopwords = [i.strip() for i in words]
    stopwords.extend(['.', '（', '）', '-'])
    out_scores = OrderedDict()
    lines = open(input_query_path, 'r', encoding='utf8').readlines()
    for line in tqdm(lines):
        corpus = []
        cpfxgc = []
        query = str(eval(line)['ridx'])
        # if query.split('.')[0] != '4738':
        #     continue
        # model init
        out_scores[query] = OrderedDict()
        result[query] = []
        files = os.listdir(os.path.join(input_candidate_path, query))
        missing_fxgc_id = []
        count = 0
        for file_ in files:
            # if file_.split('.')[0] != '11163':
            #     continue
            file_json = json.load(open(os.path.join(input_candidate_path, query, file_), 'r', encoding='utf8'))
            a = jieba.cut(file_json['ajjbqk'], cut_all=False)
            tem = " ".join(a).split()
            b = jieba.cut(file_json['ajName'], cut_all=False)
            tem_b = " ".join(b).split()
            for i in range(5):
                tem.extend(tem_b)
            corpus.append([i for i in tem if not i in stopwords])
            for word_ in corpus[count]:
                if any(word in word_ for word in ['某', '被告人']):
                    corpus[count].remove(word_)
            count += 1
            # try:
            #     b = jieba.cut(file_json['cpfxgc'], cut_all=False)
            #     tem_b = " ".join(b).split()
            # except:
            #     missing_fxgc_id.append(file_.split('.')[0])
            #     tem_b = ['空']
            # cpfxgc.append([i for i in tem_b if not i in stopwords])
        bm25Model = bm25.BM25(corpus)
        # bm25Model_key = bm25.BM25(cpfxgc)

        # rank
        a = jieba.cut(eval(line)['q'], cut_all=False)
        tem = " ".join(a).split()
        b = jieba.cut('，'.join(eval(line)['crime']), cut_all=False)
        tem_b = ' '.join(b).split()
        for i in range(5):
            tem.extend(tem_b)
        q = [i for i in tem if not i in stopwords]
        raw_scores = np.array(bm25Model.get_scores(q))
        # crime = [i for i in tem_b if not i in stopwords]
        # key_scores = np.array(bm25Model_key.get_scores(crime))
        raw_rank_index = raw_scores.argsort().tolist()[::-1]
        result[query] = [int(files[i].split('.')[0]) for i in raw_rank_index]
        temp_dict = {}
        for i in raw_rank_index:
            temp_dict[int(files[i].split('.')[0])] = raw_scores[i]
        out_scores[query] = temp_dict

    if 'test' not in input_path:
        fq = json.load(open('data/phase_2/train/label_top30_dict.json', 'r', encoding='utf8'))
        all_labels = {}
        for query_id in result.keys():
            all_labels[query_id] = fq[query_id]
        ndcg_30 = cal_ndcg(result, all_labels)
        print(ndcg_30)
        # result['ndcg30'] = ndcg_30
    # json.dump(out_scores, open(os.path.join(output_path, 'bm25_scores_train.json'), "w", encoding="utf8"), indent=2,
    #           ensure_ascii=False, sort_keys=True)
    json.dump(result, open(os.path.join(output_path, 'prediction_bm25_train.json'), "w", encoding="utf8"), indent=2,
              ensure_ascii=False, sort_keys=True)
    print('ouput done.')

