import json, os, re, jieba
from tqdm import tqdm


stop_words = set()
for w in open('./stopword.txt', encoding='utf8'):
    stop_words.add(w.strip())
stop_words = list(stop_words)
l = len(stop_words)
for c in ['！', '。', '；', '，', '、', '：', '？']:
    stop_words.remove(c)
l1 = len(stop_words)


def prepare_key_data(query_file, label_top30_file, candidate_dir, saved_file):
    queries = []
    fq = open(query_file, 'r', encoding='utf8')
    for line in fq:
        queries.append(json.loads(line.strip()))
    if label_top30_file is not None:
        fl = open(label_top30_file, 'r', encoding='utf8')
        all_label_top30 = json.load(fl)
    data = []
    for query in tqdm(queries):
        qidx, q, crime = str(query['ridx']), str(query['q']), '、'.join(query['crime'])
        # q_len = len(q)
        q = ''.join(w for w in jieba.lcut(q) if w not in stop_words)
        # f_len = len(filter_word)
        q_list = [s for s in re.split(r'[。！？]', q) if len(s) > 0]
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
            cpfxgc_first_list = cpfxgc.split('。')
            cpfxgc_list = [s for s in re.split(r'[。！；：？]', '。'.join(cpfxgc_first_list[1:])) if len(s) > 0]
            # cpfxgc_list = list(set(cpfxgc_list)).sort(key=cpfxgc_list.index)
            filter_cpfxgc = []
            for sent in cpfxgc_list:
                key_word = [['行为', '罪'], ['罪', '事实清楚']]
                or_word = ['已构成', '已经构成']
                if (all(word in sent for word in key_word[0]) or all(word in sent for word in key_word[1])) \
                        and any(word in sent for word in or_word):
                    filter_cpfxgc.append(sent)
            # last_index = cpfxgc_list.index(filter_cpfxgc[-1]) if filter_cpfxgc != [] else -1
            # rest = cpfxgc_list[:last_index+1]
            if filter_cpfxgc != []:
                filter_cpfxgc.insert(0, cpfxgc_first_list[0])
                filter_cpfxgc = '。'.join(filter_cpfxgc)
            else:
                filter_cpfxgc = cpfxgc_first_list[0] + '。'
            if label_top30_file is not None:
                label = all_label_top30[qidx][didx]/3 if didx in all_label_top30[qidx] else 0
                all_label = [qidx, didx, label]
                data.append({'query': q, 'candidate': filter_cpfxgc, 'label': all_label})
            else:
                all_label = [qidx, didx]
                data.append({'query': q, 'candidate': filter_cpfxgc, 'label': all_label})
    with open(saved_file, 'w', encoding='utf8') as fs:
        json.dump(data, fs, ensure_ascii=False, indent=2)


def prepare_case_data(query_file, label_top30_file, candidate_dir, saved_file):
    queries = []
    fq = open(query_file, 'r', encoding='utf8')
    for line in fq:
        queries.append(json.loads(line.strip()))
    if label_top30_file is not None:
        fl = open(label_top30_file, 'r', encoding='utf8')
        all_label_top30 = json.load(fl)
    data = []
    for query in tqdm(queries):
        qidx, q, crime = str(query['ridx']), str(query['q']), '、'.join(query['crime'])
        # q_len = len(q)
        q = ''.join(w for w in jieba.lcut(q) if w not in stop_words)
        # f_len = len(filter_word)
        q_list = [s for s in re.split(r'[。！？]', q) if len(s) > 0]
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
            cpfxgc_first_list = cpfxgc.split('。')
            cpfxgc_list = [s for s in re.split(r'[。！；？]', '。'.join(cpfxgc_first_list[1:])) if len(s) > 0]
            filter_cpfxgc = []
            for sent in cpfxgc_list:
                key_word = ['行为', '罪']
                or_word = ['已构成', '已经构成']
                if all(word in sent for word in key_word) and any(word in sent for word in or_word):
                    filter_cpfxgc.append(sent)
            # last_index = cpfxgc_list.index(filter_cpfxgc[-1]) if filter_cpfxgc != [] else -1
            # rest = cpfxgc_list[:last_index+1]
            if filter_cpfxgc != []:
                filter_cpfxgc.insert(0, cpfxgc_first_list[0])
                filter_cpfxgc = '。'.join(filter_cpfxgc)
            else:
                filter_cpfxgc = cpfxgc_first_list[0] + '。'
            if label_top30_file is not None:
                label = all_label_top30[qidx][didx]/3 if didx in all_label_top30[qidx] else 0
                all_label = [qidx, didx, label]
                data.append({'query': q, 'candidate': filter_cpfxgc, 'label': all_label})
            else:
                all_label = [qidx, didx]
                data.append({'query': q, 'candidate': filter_cpfxgc, 'label': all_label})
    with open(saved_file, 'w', encoding='utf8') as fs:
        json.dump(data, fs, ensure_ascii=False, indent=2)


if __name__ == "__main__":

    query_file = './data/phase_1/train/query.json'
    label_top30_file = './data/phase_1/train/label_top30_dict.json'
    candidate_dir = './data/phase_1/train/candidates/'
    saved_file = 'data/case_dev.json'
    prepare_key_data(query_file, label_top30_file, candidate_dir, saved_file)
    prepare_case_data(query_file, label_top30_file, candidate_dir, saved_file)

    query_file = './data/phase_2/train/query.json'
    label_top30_file = './data/phase_2/train/label_top30_dict.json'
    candidate_dir = './data/phase_2/train/candidates/'
    saved_file = 'data/case_train.json'
    prepare_key_data(query_file, label_top30_file, candidate_dir, saved_file)
    prepare_case_data(query_file, label_top30_file, candidate_dir, saved_file)
    pass
