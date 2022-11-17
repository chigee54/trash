import json, os, re, jieba
from tqdm import tqdm
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
    cpfxgc_list = [s for s in re.split(r'[。；]', '。'.join(cpfxgc_first_list[1:])) if len(s) > 0]
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
        filter_cpfxgc = cpfxgc_first_list[0]
    # filter_cpfxgc_list = filter_cpfxgc.split('，')
    # for comma_sent in filter_cpfxgc_list:
    #     key_word = [['行为', '罪'], ['罪', '事实清楚']]
    #     if all(word in comma_sent for word in key_word[0]) or all(word in comma_sent for word in key_word[1]):
    #         filter_cpfxgc_list.remove(comma_sent)
    # cpfxgc = '，'.join(filter_cpfxgc_list)
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


def extract_matching(texts, candidate, crime, topk=None):
    if topk:
        ids = np.argsort([compute_main_metric(t, candidate, 'char') for t in texts])
        filter_query = '；'.join([texts[id] for id in sorted(ids[:topk])])
    else:
        id = np.argmax([compute_main_metric(t, candidate, 'char') for t in texts])
        filter_query = texts[id] + '。其行为均已构成' + crime
    return filter_query


def query_strategy(q, crime, flag=1):
    q = ''.join(w for w in jieba.lcut(q) if w not in stop_words)
    q_list = [s for s in re.split(r'[。；]', q) if len(s) > 0]
    if flag == 1:
        start, overlap = 0, 2
        window_size = 5
        update_q_list = []
        while start < len(q_list):
            window_tokens = q_list[start:start + window_size]
            if len(window_tokens) > 3:
                update_q_list.append('。'.join(window_tokens))
            else:
                update_q_list.append('。'.join(window_tokens))
            start += overlap
        q = update_q_list
    elif flag == 2:
        q_sent_num = len(q_list)
        if q_sent_num < 6:
            q = [q + '其行为均已构成' + crime]
        elif q_sent_num < 11:
            step = q_sent_num // 2
            q = ['。'.join(q_list[:step]) + '。其行为均已构成' + crime,
                 '。'.join(q_list[step:]) + '。其行为均已构成' + crime]
        elif q_sent_num < 16:
            step = q_sent_num // 3
            q = ['。'.join(q_list[:step]) + '。其行为均已构成' + crime,
                 '。'.join(q_list[step:step * 2]) + '。其行为均已构成' + crime,
                 '。'.join(q_list[step * 2:]) + '。其行为均已构成' + crime]
        else:
            step = q_sent_num // 4
            q = ['。'.join(q_list[:step]) + '。其行为均已构成' + crime,
                 '。'.join(q_list[step:step * 2]) + '。其行为均已构成' + crime,
                 '。'.join(q_list[step * 2:step * 3]) + '。其行为均已构成' + crime]
    elif flag == 3:
        q_sent_num = len(q_list)
        if q_sent_num < 6:
            q = [q]
        elif q_sent_num < 11:
            step = q_sent_num // 2
            q = ['。'.join(q_list[:step]),
                 '。'.join(q_list[step:])]
        elif q_sent_num < 16:
            step = q_sent_num // 3
            q = ['。'.join(q_list[:step]),
                 '。'.join(q_list[step:step * 2]),
                 '。'.join(q_list[step * 2:])]
        else:
            step = q_sent_num // 4
            q = ['。'.join(q_list[:step]),
                 '。'.join(q_list[step:step * 2]),
                 '。'.join(q_list[step * 2:step * 3])]
    elif flag == 4:
        q = q_list
    return q


def prepare_case_data(query_file, candidate_dir, saved_file, label_top30_file):
    queries = []
    fq = open(query_file, 'r', encoding='utf8')
    for line in fq:
        queries.append(json.loads(line.strip()))
    all_label_top30 = json.load(open(label_top30_file, 'r', encoding='utf8')) if label_top30_file else None
    data = []
    for query in tqdm(queries, desc=u'案情抽取'):
        qidx, q, crime = str(query['ridx']), str(query['q']), '、'.join(query['crime'])
        doc_dir = os.path.join(candidate_dir, qidx)
        doc_files = os.listdir(doc_dir)
        for doc_file in doc_files:
            doc_path = os.path.join(doc_dir, doc_file)
            didx = str(doc_file.split('.')[0])
            with open(doc_path, 'r', encoding='utf8') as fd:
                sample_d = json.load(fd)
            try:
                doc, cpfxgc = sample_d['ajjbqk'], sample_d['cpfxgc']
            except:
                continue
            filter_cpfxgc = cpfxgc_filter(cpfxgc)
            query_list = query_strategy(q, crime)
            # if len(query_list) == 0 or len(filter_cpfxgc) == 0:
            #     continue
            filter_query = extract_matching(query_list, filter_cpfxgc, crime)
            if label_top30_file:
                label = all_label_top30[qidx][didx]/3 if didx in all_label_top30[qidx] else 0
                all_label = [qidx, didx, label]
                data.append({'query': filter_query, 'candidate': filter_cpfxgc, 'label': all_label})
            else:
                all_label = [qidx, didx]
                data.append({'query': filter_query, 'candidate': filter_cpfxgc, 'label': all_label})
    with open(saved_file, 'w', encoding='utf8') as fs:
        json.dump(data, fs, ensure_ascii=False, indent=2)


def merge_data():
    with open('data/extract_data/extract_large.json', 'r', encoding='utf8') as f1:
        a = json.load(f1)
    with open('data/extract_data/extract_small.json', 'r', encoding='utf8') as f2:
        b = json.load(f2)
    a.extend(b)
    with open('data/extract_data/extract_all.json', 'w', encoding='utf8') as fs:
        json.dump(c, fs, ensure_ascii=False, indent=2)


if __name__ == "__main__":

    # query_file = './data/phase_2/train/query.json'
    # candidate_dir = './data/phase_2/train/candidates/'
    # label_top30_file = './data/phase_2/train/label_top30_dict.json'
    # saved_file = 'data/case2_train.json'
    # prepare_case_data(query_file, candidate_dir, saved_file, label_top30_file)

    query_file = './data/phase_2/test/query.json'
    candidate_dir = './data/phase_2/test/candidates/'
    label_top30_file = './data/phase_2/test/label_top30_dict.json'
    saved_file = 'data/case2_test.json'
    prepare_case_data(query_file, candidate_dir, saved_file, None)

    # merge_data()
    pass
