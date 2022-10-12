import argparse
import json
from tqdm import tqdm
import os, time, logging
import torch
from torch.utils.data import DataLoader
from long_train import LongModel, LongDataset, _move_to_device

parser = argparse.ArgumentParser(description="Help info.")
parser.add_argument('--input', type=str, default='./data/phase_2/test/', help='input path of the dataset directory.')
parser.add_argument('--output', type=str, default='./result/', help='output path of the prediction file.')

args = parser.parse_args()
input_path = args.input
input_query_path = os.path.join(input_path, 'query.json')
input_candidate_path = os.path.join(input_path, 'candidates')
output_path = args.output
new_data_path = os.path.join(os.path.dirname(__file__), 'new_test.json')


def convert_data(query_file, candidate_dir, saved_file):
    queries = []
    fq = open(query_file, 'r', encoding='utf8')
    for line in fq:
        queries.append(json.loads(line.strip()))
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
            all_label = [qidx, didx]
            data.append({'query': q, 'candidate': doc, 'label': all_label})
    with open(saved_file, 'w', encoding='utf8') as fs:
        json.dump(data, fs, ensure_ascii=False, indent=2)


def predict():
    model = LongModel('../pretrained_model/lawformer').cuda()
    model.load_state_dict(torch.load('result/bsz-1-lr-1e-05/2epoch_0.939_t_bert.pt'))
    dataloader = DataLoader(LongDataset('data/textrank_filter/test.json'), batch_size=18, shuffle=False)
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
    json.dump(result, open(os.path.join(output_path, 'prediction.json'), "w", encoding="utf8"), indent=2,
              ensure_ascii=False, sort_keys=True)
    print('output done.')


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    main()
