import json, os, re, jieba
from tqdm import tqdm
from gensim.summarization import bm25
from sklearn import preprocessing
import numpy as np
from rouge import Rouge
import sys
sys.setrecursionlimit(4000)

stop_words = set()
for w in open('./stopword.txt', encoding='utf8'):
    stop_words.add(w.strip())
stop_words = list(stop_words)
for c in ['！', '。', '；', '，', '、', '：', '？']:
    stop_words.remove(c)


def cpfxgc_filter(cpfxgc):
    cpfxgc_first_list = cpfxgc.split('。')
    cpfxgc_list = [s for s in re.split(r'[。！；？]', '。'.join(cpfxgc_first_list[1:])) if len(s) > 0]
    filter_cpfxgc = []
    for sent in cpfxgc_list:
        key_word = [['行为', '罪'], ['罪', '事实清楚']]
        or_word = ['已构成', '已经构成']
        if (all(word in sent for word in key_word[0]) or all(word in sent for word in key_word[1])) \
                and any(word in sent for word in or_word):
            filter_cpfxgc.append(sent)
    if filter_cpfxgc != []:
        filter_cpfxgc.insert(0, cpfxgc_first_list[0])
        filter_cpfxgc = '。'.join(filter_cpfxgc)
    else:
        filter_cpfxgc = cpfxgc_first_list[0] + '。'
    filter_cpfxgc_list = filter_cpfxgc.split('，')
    for comma_sent in filter_cpfxgc_list:
        key_word = [['行为', '罪'], ['罪', '事实清楚']]
        if all(word in comma_sent for word in key_word[0]) or all(word in comma_sent for word in key_word[1]):
            filter_cpfxgc_list.remove(comma_sent)
    cpfxgc = '，'.join(filter_cpfxgc_list)
    cpfxgc = string_rule(cpfxgc)
    return cpfxgc


def string_rule(string):
    p = re.compile(r'[（](.*?)[）]', re.S)
    b = re.compile(r'(被告人)(.*?)(等人)', re.S)
    string = re.sub(b, '', string)
    string = re.sub(p, '', string)
    string = string.replace('被告人', '')
    string = string.replace('本院认为', '')
    return string


def compute_main_metric(source, target, unit='word'):
    """计算主要metric
    """
    if unit == 'word':
        source = jieba.cut(source, HMM=False)
        target = jieba.cut(target, HMM=False)
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = Rouge().get_scores(hyps=source, refs=target)
        metrics = {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
        metrics['main'] = (metrics['rouge-1'] * 0.2 +
                           metrics['rouge-2'] * 0.4 +
                           metrics['rouge-l'] * 0.4)
        return metrics['main']
    except ValueError:
        metrics = {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
            'main': 0.0
        }
        return metrics['main']


def extract_matching(texts, summaries):
    ids = []
    for summary in summaries:
        # id = np.argmax([compute_main_metric(t, summary, 'char') for t in texts])
        id = np.argsort([compute_main_metric(t, summary, 'char') for t in texts])[:3]
        ids.extend(id)
    labels = sorted(set(i for i in ids))
    pre_event = '；'.join([texts[i] for i in labels])
    return labels, pre_event


def prepare_case_data(query_file, candidate_dir, saved_file):
    queries = []
    fq = open(query_file, 'r', encoding='utf8')
    for line in fq:
        queries.append(json.loads(line.strip()))
    data = []
    for query in tqdm(queries, desc=u'案情抽取'):
        qidx, q, crime = str(query['ridx']), str(query['q']), '、'.join(query['crime'])
        doc_dir = os.path.join(candidate_dir, qidx)
        doc_files = os.listdir(doc_dir)
        for doc_file in doc_files:
            doc_path = os.path.join(doc_dir, doc_file)
            with open(doc_path, 'r', encoding='utf8') as fd:
                sample_d = json.load(fd)
            try:
                ajjbqk, cpfxgc = sample_d['ajjbqk'], sample_d['cpfxgc']
            except:
                continue
            filter_cpfxgc = cpfxgc_filter(cpfxgc)
            ajjbqk = list(set([s for s in re.split(r'[。，：；]', ajjbqk[:1200]) if len(s) > 0]))
            filter_cpfxgc = list(set([s for s in re.split(r'[。，：；]', filter_cpfxgc) if len(s) > 0]))
            if len(ajjbqk) < 2 or len(filter_cpfxgc) == 0:
                continue
            labels, pre_event = extract_matching(ajjbqk, filter_cpfxgc)
            data.append({'ajjbqk': ajjbqk, 'labels': labels})
    with open(saved_file, 'w', encoding='utf8') as fs:
        json.dump(data, fs, ensure_ascii=False, cls=NpEncoder)


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


def merge_data():
    with open('data/extract_data/extract_large.json', 'r', encoding='utf8') as f1:
        a = json.load(f1)
    with open('data/extract_data/extract_small.json', 'r', encoding='utf8') as f2:
        b = json.load(f2)
    a.extend(b)
    with open('data/extract_data/extract_all.json', 'w', encoding='utf8') as fs:
        json.dump(c, fs, ensure_ascii=False, cls=NpEncoder)


if __name__ == "__main__":

    # query_file = './data/phase_1/train/query.json'
    # candidate_dir = './data/phase_1/train/candidates/'
    # saved_file = 'data/extract_data/extract_small.json'
    # prepare_case_data(query_file, candidate_dir, saved_file)

    query_file = './data/phase_2/train/query.json'
    candidate_dir = './data/phase_2/train/candidates/'
    saved_file = 'data/extract_data/extract_large.json'
    prepare_case_data(query_file, candidate_dir, saved_file)

    # merge_data()
    pass
