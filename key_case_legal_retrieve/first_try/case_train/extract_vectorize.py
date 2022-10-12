import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json, re
from tqdm import tqdm
from new_utils import sequence_padding


class GlobalAveragePooling1D(nn.Module):
    """自定义全局池化
    对一个句子的pooler取平均，一个长句子用短句的pooler平均代替
    """
    def __init__(self):
        super(GlobalAveragePooling1D, self).__init__()

    def forward(self, inputs, mask=None):
        if mask is not None:
            mask = mask.to(torch.float)[:, :, None]
            return torch.sum(inputs * mask, dim=1) / torch.sum(mask, dim=1)
        else:
            return torch.mean(inputs, dim=1)


class MLP(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)
        return x


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


class LegalRoBerta(nn.Module):
    def __init__(self, model_path):
        super(LegalRoBerta, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.BatchMLP = BatchNormMLP()
        self.MLP = MLP()

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.model(input_ids, attention_mask, token_type_ids)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state


class LoadDataset(Dataset):
    def __init__(self, data_file, model_path, save_json, type):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.data = self.load_data(data_file, save_json, type)

    def __getitem__(self, index):
        return self.data[index]

    def load_data(self, data_file, save_json, type):
        if type == 'test':
        ###############################################
            all_data = data_file
        ###############################################
        else:
            all_data = json.load(open(data_file, 'r', encoding='utf8'))
        ###############################################
        query_list = []
        feature_list = []
        for data in all_data:
            if type != 'test':
            ########################################################
                text = data['ajjbqk']
                sent_list = [sent.strip() for sent in text]
                if len(sent_list) == 0:
                    continue
                query_list.append({'query': sent_list})
            ########################################################
            else:
                text, qidx = str(data['q']), str(data['ridx'])
                sent_list = list(set([s.strip() for s in re.split(r'[。，：；]', text) if len(s) > 1]))
                if len(sent_list) == 0:
                    continue
                query_list.append({'qidx': qidx, 'query': sent_list})
            ########################################################
            feature = self.tokenizer(sent_list, max_length=16,
                                     truncation=True, padding='max_length', return_tensors='pt')
            feature_list.append(feature)
        if save_json is not None:
            with open(save_json, 'w', encoding='utf8') as fs:
                json.dump(query_list, fs, ensure_ascii=False, indent=2)
        return feature_list

    def __len__(self):
        return len(self.data)


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


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model_path = "../pretrained_model/legal_roberta"
    data_type = 'test'

    if data_type != 'test':
    ##################################################################
        data_file = './data/extract_data/extract_all.json'
        data_extract_npy = './data/extract_data/large_extract_all'
        # data_extract_npy = './data/extract_data/small_extract'
        dataset_ = LoadDataset(data_file, model_path, None, data_type)
    ##################################################################
    else:
        # data_file = './data/phase_2/test/query.json'
        # save_json = './data/test/test_query_extract.json'
        # data_extract_npy = './data/test/test_query_extract'
        data_file = './data/phase_1/train/query.json'
        save_json = './data/query_convert/small_query_extract.json'
        data_extract_npy = './data/query_convert/small_query_extract'
        queries = []
        fq = open(data_file, 'r', encoding='utf8')
        for line in fq:
            queries.append(json.loads(line.strip()))
        dataset_ = LoadDataset(queries, model_path, save_json, data_type)
    ##################################################################
    model = LegalRoBerta(model_path).cuda()
    dataset = DataLoader(dataset_, batch_size=1, shuffle=False)
    all_embeddings = []
    i = 0
    for data in tqdm(dataset, desc='数据转换'):
        # i += 1
        # if i == 3:
        #     break
        for k, v in data.items():
            data[k] = v.squeeze(0).cuda()
        if data['input_ids'].shape[0] > 50:
            data_ = {}
            for k, v in data.items():
                data_[k] = v[:50]
            embeddings = model(**data_)
            for k, v in data.items():
                data[k] = v[50:]
            last_embeddings = model(**data)
            embeddings = torch.cat([embeddings, last_embeddings], dim=0)
            embeddings = torch.mean(embeddings, dim=1)
            all_embeddings.append(embeddings.cpu().detach().numpy())
        else:
            embeddings = model(**data)
            embeddings = torch.mean(embeddings, dim=1)
            all_embeddings.append(embeddings.cpu().detach().numpy())
    all_embeddings = sequence_padding(all_embeddings)
    np.save(data_extract_npy, all_embeddings)
    print(u'输出路径：%s' % data_extract_npy)
