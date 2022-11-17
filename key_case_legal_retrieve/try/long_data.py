import json, os
from tqdm import tqdm


def prepare_data(query_file, label_top30_file, candidate_dir, saved_file):
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

        doc_dir = os.path.join(candidate_dir, qidx)
        doc_files = os.listdir(doc_dir)
        for doc_file in doc_files:
            doc_path = os.path.join(doc_dir, doc_file)
            didx = str(doc_file.split('.')[0])
            with open(doc_path, 'r', encoding='utf8') as fd:
                sample_d = json.load(fd)
            try:
                doc, d_crime = sample_d['ajjbqk'], sample_d['ajName']
            except:
                continue
            if label_top30_file is not None:
                label = all_label_top30[qidx][didx]/3 if didx in all_label_top30[qidx] else 0
                all_label = [qidx, didx, label]
                data.append({'query': q, 'candidate': doc, 'label': all_label})
            else:
                data.append({'query': q, 'candidate': doc})
    with open(saved_file, 'w', encoding='utf8') as fs:
        json.dump(data, fs, ensure_ascii=False, indent=2)


if __name__ == "__main__":

    query_file = './data/phase_1/train/query.json'
    label_top30_file = './data/phase_1/train/label_top30_dict.json'
    candidate_dir = './data/phase_1/train/candidates/'
    saved_file = 'data/new_dev.json'
    prepare_data(query_file, label_top30_file, candidate_dir, saved_file)

    query_file = './data/phase_2/train/query.json'
    label_top30_file = './data/phase_2/train/label_top30_dict.json'
    candidate_dir = './data/phase_2/train/candidates/'
    saved_file = 'data/new_train.json'
    prepare_data(query_file, label_top30_file, candidate_dir, saved_file)
    pass
