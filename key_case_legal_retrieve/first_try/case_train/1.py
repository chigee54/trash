import json
import numpy as np
from numpy import mean


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


# train_npy = np.load('data/large_extract.npy')[:, :101, :]
# dev_npy = np.load('data/small_extract.npy')
# new_npy = np.concatenate((train_npy, dev_npy), axis=0)
# np.save('data/large_extract_all', new_npy)

f1 = open('data/extract_data/extract_large.json', 'r', encoding='utf8')
a = json.load(f1)
f2 = open('data/extract_data/extract_small.json', 'r', encoding='utf8')
b = json.load(f2)
c = a
c.extend(b)
with open('data/extract_data/extract_all.json', 'w', encoding='utf8') as fs:
    json.dump(c, fs, ensure_ascii=False, cls=NpEncoder)

# ranks = [1, 3, 0, 2, 1, 0]
# new_ranks = [i - 2 if i > 1 else i for i in ranks]
# print(new_ranks)
# with open('data/query_convert/case_dev.json', 'r', encoding='utf8') as fa:
#     data = json.load(fa)
#     for i in data:
#         ori_label = i['label'][2] * 3
#         if ori_label > 1:
#             i['label'][2] = int(ori_label - 2)
#         else:
#             i['label'][2] = int(ori_label)
#         update_label.append(i)
# with open('data/extract_all.json', 'w', encoding='utf8') as fs:
#     json.dump(c, fs, ensure_ascii=False, cls=NpEncoder)


#     for i in data:
#         length = len(i['query'])
#         length_ = len(i['candidate'])
#         all_len.append(length)
#         all_candi_len.append(length_)
#
# print('query length min: {}'.format(min(all_len)))
# print('query length mean: {}'.format(mean(all_len)))
# print('query length max: {}'.format(max(all_len)))
#
# print('candidate length min: {}'.format(min(all_candi_len)))
# print('candidate length mean: {}'.format(mean(all_candi_len)))
# print('candidate length max: {}'.format(max(all_candi_len)))


# all_labels = []
# all_text_len = []
# all_sent_len = []
# all_labels_len = []
# with open('./data/extract_large.json', 'r', encoding='utf8') as ff:
#     for line in json.load(ff):
#         labels = line['labels']
#         leng = len(line['ajjbqk'])
#         for sent in line['ajjbqk']:
#             sent_len = len(sent)
#             all_sent_len.append(sent_len)
#         all_text_len.append(leng)
#         all_labels.extend(labels)
#         all_labels_len.append(len(labels))
#
# print('Labels number min: {}'.format(min(all_labels)))
# print('Labels number mean: {}'.format(mean(all_labels)))
# print('Labels number max: {}'.format(max(all_labels)))
#
# print('Texts length min: {}'.format(min(all_text_len)))
# print('Texts length mean: {}'.format(mean(all_text_len)))
# print('Texts length max: {}'.format(max(all_text_len)))
#
# print('Sentence length min: {}'.format(min(all_sent_len)))
# print('Sentence length mean: {}'.format(mean(all_sent_len)))
# print('Sentence length max: {}'.format(max(all_sent_len)))
#
# print('Labels length min: {}'.format(min(all_labels_len)))
# print('Labels length mean: {}'.format(mean(all_labels_len)))
# print('Labels length max: {}'.format(max(all_labels_len)))