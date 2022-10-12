import argparse
import json
from tqdm import tqdm
import os, time, logging, re, jieba
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset


parser = argparse.ArgumentParser(description="Help info.")
parser.add_argument('--input', type=str, default='./data/phase_2/test/', help='input path of the dataset directory.')
parser.add_argument('--output', type=str, default='./result/', help='output path of the prediction file.')

args = parser.parse_args()
input_path = args.input
input_query_path = os.path.join(input_path, 'query.json')
input_candidate_path = os.path.join(input_path, 'candidates')
output_path = args.output
key_data_path = os.path.join(os.path.dirname(__file__), 'data/test/key_test.json')
case_data_path = os.path.join(os.path.dirname(__file__), 'data/test/case_test.json')


stop_words = set()
for w in open('./stopword.txt', encoding='utf8'):
    stop_words.add(w.strip())
stop_words = list(stop_words)
for c in ['！', '。', '；', '，', '、', '：', '？']:
    stop_words.remove(c)


class LegalModel(nn.Module):
    def __init__(self, model_path):
        super(LegalModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.linear = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, label=None):
        output = self.model(input_ids, attention_mask, token_type_ids)
        logits = self.linear(output[1])
        score = torch.sigmoid(logits).squeeze(-1)
        return score, label


class CreateDataset(Dataset):
    def __init__(self, data_file, data_type):
        self.tokenizer = AutoTokenizer.from_pretrained("../pretrained_model/legal_roberta")
        self.data_type = data_type
        with open(data_file, 'r', encoding='utf8') as f:
            self.data = json.load(f)

    def __getitem__(self, index):
        data = self.data[index]
        if self.data_type == 'case':
            query = data['query']
        else:
            query = data['query'][0] if len(data['query']) == 1 else data['query'][0]
        query_cut = self.tokenizer.tokenize(query)
        candidate_cut = self.tokenizer.tokenize(data['candidate'])
        total_length = len(query_cut) + len(candidate_cut)
        if total_length > 509:
            for _ in range(total_length - 509):
                if len(query_cut) > len(candidate_cut):
                    query_cut = query_cut[1:]
                else:
                    candidate_cut = candidate_cut[1:]
        tokens = ['[CLS]'] + query_cut + ['[SEP]'] + candidate_cut + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        types = [0] * (len(query_cut) + 2) + [1] * (len(candidate_cut) + 1)
        input_ids, token_type_ids, attention_mask = self.pad_seq(token_ids, types)
        feature = {'input_ids': torch.LongTensor(input_ids),
                   'token_type_ids': torch.LongTensor(token_type_ids),
                   'attention_mask': torch.LongTensor(attention_mask),
                   'label': data['label']}
        return feature

    def pad_seq(self, ids, types):
        batch_len = 512
        masks = [1] * len(ids) + [0] * (batch_len - len(ids))
        types += [0] * (batch_len - len(ids))
        ids += [0] * (batch_len - len(ids))
        return ids, types, masks

    def __len__(self):
        return len(self.data)


def convert_key_data(query_file, candidate_dir, saved_file):
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


def convert_case_data(query_file, candidate_dir, saved_file):
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


def _move_to_device(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()
    return batch


def predict(gama):
    case_model = LegalModel('../pretrained_model/legal_roberta').cuda()
    case_model.load_state_dict(torch.load('result/bsz-24-lr-1e-05/10epoch_0.9838_bert.pt'))
    case_dataloader = DataLoader(CreateDataset(case_data_path, 'case'), batch_size=128, shuffle=False)
    case_model.eval()
    key_model = LegalModel('../pretrained_model/legal_roberta').cuda()
    key_model.load_state_dict(torch.load('result/case/seed1-bsz24-lr1e-05/bert.pt'))  # 0.9280
    key_dataloader = DataLoader(CreateDataset(key_data_path, 'key'), batch_size=128, shuffle=False)
    key_model.eval()
    with torch.no_grad():
        all_preds, info = {}, {}
        for key_data, case_data in tqdm(zip(key_dataloader, case_dataloader)):
            key_data = _move_to_device(key_data)
            key_score, docids = key_model(**key_data)
            case_data = _move_to_device(case_data)
            case_score, _ = key_model(**case_data)
            score = gama * key_score + (1 - gama) * case_score
            for n, (qid, cid) in enumerate(zip(docids[0], docids[1])):
                if qid not in info.keys():
                    info[qid] = [[cid], [score[n]]]
                else:
                    info[qid][1].append(score[n])
                    info[qid][0].append(cid)
        for qidx, (dids, preds) in info.items():
            sorted_r = sorted(list(zip(dids, preds)), key=lambda x: x[1], reverse=True)
            pred_ids = [x[0] for x in sorted_r]
            all_preds[qidx] = pred_ids
    return all_preds


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S')
    print('begin...')
    if not os.path.exists(key_data_path):
        convert_key_data(input_query_path, input_candidate_path, key_data_path)

    if not os.path.exists(case_data_path):
        convert_case_data(input_query_path, input_candidate_path, case_data_path)
    time.sleep(1)
    print('temp data converting finished...')

    print('prediction starting...')
    result = predict(0.7)
    json.dump(result, open(os.path.join(output_path, 'prediction.json'), "w", encoding="utf8"), indent=2,
              ensure_ascii=False, sort_keys=True)
    print('output done.')


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    main()
