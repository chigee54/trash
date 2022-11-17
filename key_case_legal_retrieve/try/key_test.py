import argparse
import json
from tqdm import tqdm
import os, time, logging, re, jieba
import torch
from torch.utils.data import DataLoader
from key_train import LongModel, LongDataset, _move_to_device

parser = argparse.ArgumentParser(description="Help info.")
parser.add_argument('--input', type=str, default='./data/phase_2/test/', help='input path of the dataset directory.')
parser.add_argument('--output', type=str, default='./result/', help='output path of the prediction file.')

args = parser.parse_args()
input_path = args.input
input_query_path = os.path.join(input_path, 'query.json')
input_candidate_path = os.path.join(input_path, 'candidates')
output_path = args.output
new_data_path = os.path.join(os.path.dirname(__file__), 'data/key_dev.json')


stop_words = set()
for w in open('./stopword.txt', encoding='utf8'):
    stop_words.add(w.strip())
stop_words = list(stop_words)
for c in ['！', '。', '；', '，', '、', '：', '？']:
    stop_words.remove(c)


def convert_data(query_file, candidate_dir, saved_file):
    queries = []
    fq = open(query_file, 'r', encoding='utf8')
    for line in fq:
        queries.append(json.loads(line.strip()))
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
            cpfxgc_list = [s for s in re.split(r'[。！；？]', '。'.join(cpfxgc_first_list[1:])) if len(s) > 0]
            # cpfxgc_list = list(set(cpfxgc_list)).sort(key=cpfxgc_list.index)
            filter_cpfxgc = []
            for sent in cpfxgc_list:
                key_word = ['行为', '罪']
                or_word = ['已构成', '已经构成']
                if all(word in sent for word in key_word) and any(word in sent for word in or_word):
                    filter_cpfxgc.append(sent)
            if filter_cpfxgc != []:
                filter_cpfxgc.insert(0, cpfxgc_first_list[0])
                filter_cpfxgc = '。'.join(filter_cpfxgc)
            else:
                filter_cpfxgc = cpfxgc_first_list[0] + '。'
            all_label = [qidx, didx]
            data.append({'query': q, 'candidate': filter_cpfxgc, 'label': all_label})
    with open(saved_file, 'w', encoding='utf8') as fs:
        json.dump(data, fs, ensure_ascii=False, indent=2)


from key_train import cal_ndcg
def predict():
    model = LongModel('../pretrained_model/legal_roberta').cuda()
    # all_labels = json.load(open('data/phase_1/train/label_top30_dict.json', 'r', encoding='utf8'))
    model.load_state_dict(torch.load('result/bsz-24-lr-1e-05/7epoch_0.9431_bert.pt'))  # 7epoch_0.9431
    # model.load_state_dict(torch.load('result/bsz-24-lr-1e-05/10epoch_0.9838_bert.pt'))
    filter_ids = {}
    with open('result/prediction_bm25_dev_4.json', 'r', encoding='utf8') as fp:
        bm25_prediction = json.load(fp)
        for k, v in bm25_prediction.items():
            v = v[50:]
            filter_ids[k] = v
    dataloader = DataLoader(LongDataset(new_data_path, None, filter_ids), batch_size=128, shuffle=False)
    model.eval()
    with torch.no_grad():
        all_preds, info = {}, {}
        for data in tqdm(dataloader):
            data = _move_to_device(data)
            score, label = model(**data)
            for n, i in enumerate(zip(label[0], label[1])):
                if i[0] not in info.keys():
                    info[i[0]] = [[i[1]], [score[n]]]
                else:
                    info[i[0]][1].append(score[n])
                    info[i[0]][0].append(i[1])
        for qidx in info.keys():
            dids, preds = info[qidx]
            sorted_r = sorted(list(zip(dids, preds)), key=lambda x: x[1], reverse=True)
            pred_ids = [x[0] for x in sorted_r]
            all_preds[qidx] = pred_ids
    # ndcg_30 = cal_ndcg(all_preds, all_labels)
    # print(ndcg_30)
    return all_preds


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S')
    print('begin...')
    if not os.path.exists(new_data_path):
        convert_data(input_query_path, input_candidate_path, new_data_path)
    time.sleep(1)
    print('temp data converting finished...')

    print('prediction starting...')
    result = predict()
    json.dump(result, open(os.path.join(output_path, 'prediction_dev_4.json'), "w", encoding="utf8"), indent=2,
              ensure_ascii=False, sort_keys=True)
    print('output done.')


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    main()
