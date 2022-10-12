import pickle

from tqdm import tqdm
import json
import os
import time
import math
from loguru import logger
import numpy as np
from os.path import join
import torch
import random
import argparse
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup


def main(seed):
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='cuda', type=str)
    parser.add_argument('-seed', default=seed, type=int)
    parser.add_argument('-label_file', default='./data/phase_1/train/label_top30_dict.json', type=str)
    parser.add_argument('-train_file', default='data/query_convert/case_train.json', type=str)
    parser.add_argument('-dev_file', default='data/query_convert/case_dev.json', type=str)
    parser.add_argument('-model_path', default='../pretrained_model/legal_roberta', type=str)
    parser.add_argument('-output_path', default='./result/case', type=str)
    parser.add_argument('-report_step', default=100, type=int)
    parser.add_argument('-early_stop', default=1000, type=int)
    parser.add_argument('-max_length', default=512, type=int)
    parser.add_argument('-eval_batch_size', default=128, type=int)
    parser.add_argument('-batch_size', default=24, type=int)
    parser.add_argument('-num_epochs', default=20, type=int)
    parser.add_argument('-learning_rate', default=1e-5, type=float)
    parser.add_argument('-warm_up_proportion', default=0, type=float)
    parser.add_argument('-gradient_accumulation_step', default=1, type=int)
    args = parser.parse_args()

    # config environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)
    args.output_path = join(args.output_path, 'seed{}-bsz{}-lr{}'.format(args.seed, args.batch_size, args.learning_rate))
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    cur_time = time.strftime("%Y%m%d_%H_%M", time.localtime())
    logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
    logger.info(args)

    model = LongModel(args.model_path).cuda()
    # model = nn.DataParallel(model, device_ids=[0, 1], output_device=0)
    start_time = time.time()
    train(model, args)
    logger.info('run time: {:.2f}'.format((time.time() - start_time) / 60))


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class LongModel(nn.Module):
    def __init__(self, model_path):
        super(LongModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.linear = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, label=None):
        output = self.model(input_ids, attention_mask, token_type_ids)
        logits = self.linear(output[1])
        score = torch.sigmoid(logits).squeeze(-1)
        return score, label


class CaseDataset(Dataset):
    def __init__(self, data_file):
        self.tokenizer = AutoTokenizer.from_pretrained("../pretrained_model/legal_roberta")
        if data_file is not None:
            with open(data_file, 'r', encoding='utf8') as f:
                self.data = json.load(f)
        else:
            self.data = None

    def __getitem__(self, index):
        data = self.data[index]
        query = data['query']
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


def scheduler_with_optimizer(model, train_loader, args):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) * args.num_epochs * args.warm_up_proportion // args.gradient_accumulation_step,
        num_training_steps=len(train_loader) * args.num_epochs // args.gradient_accumulation_step)
    return optimizer, scheduler


def cosent(y_pred, y_true):
    y_pred = y_pred * 20
    y_pred = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)  # 这里是算出所有位置 两两之间余弦的差值
    y_true = y_true.unsqueeze(1) < y_true.unsqueeze(0)  # 取出负例-正例的差值
    y_true = y_true.float()
    y_pred = y_pred - (1 - y_true) * 1e12
    y_pred = y_pred.view(-1)
    y_pred = torch.cat((torch.tensor([0]).float().cuda(), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
    return torch.logsumexp(y_pred, dim=0)


def _move_to_device(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()
    return batch


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
        ranks = [did2rel[idx] if idx in did2rel else 0 for idx in pred_ids]
        ranks = [i - 2 if i > 1 else i for i in ranks]
        ndcgs.append(ndcg(ranks, 30))
        # print(f'********** qidx: {qidx} **********')
        # print(f'top30 pred_ids: {pred_ids}')
        # print(f'ranks: {ranks}')
    print(ndcgs)
    return sum(ndcgs) / len(ndcgs)


def evaluate(model, dataloader, all_labels):
    model.eval()
    with torch.no_grad():
        all_preds, info = {}, {}
        for data in dataloader:
            data = _move_to_device(data)
            score, label = model(**data)
            for n, i in enumerate(zip(label[0], label[1], label[2])):
                if i[0] not in info.keys():
                    info[i[0]] = [[i[1]], [score[n]]]
                else:
                    info[i[0]][1].append(score[n])
                    info[i[0]][0].append(i[1])
        for qidx in info.keys():
            # max_len = max((len(l) for l in info[qidx]))
            # pad_length = list(map(lambda l: l + [0] * (max_len - len(l)), info[qidx]))
            dids, preds = info[qidx]
            sorted_r = sorted(list(zip(dids, preds)), key=lambda x: x[1], reverse=True)
            pred_ids = [x[0] for x in sorted_r]
            all_preds[qidx] = pred_ids[:30]
    ndcg_30 = cal_ndcg(all_preds, all_labels)
    del info, all_preds
    torch.cuda.empty_cache()
    return ndcg_30


def train(model, args):
    all_labels = json.load(open(args.label_file, 'r', encoding='utf8'))
    train_data = CaseDataset(args.train_file)
    dev_data = CaseDataset(args.dev_file)
    dev_loader = DataLoader(dev_data, batch_size=args.eval_batch_size, shuffle=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    optimizer, scheduler = scheduler_with_optimizer(model, train_loader, args)
    best = 0
    for epoch in range(args.num_epochs):
        # model.load_state_dict(torch.load('result/output/bsz-1-lr-1e-05/epoch2_0.928.pt'))
        torch.cuda.empty_cache()
        model.train()
        model.zero_grad()
        early_stop = 0
        for batch_idx, data in enumerate(tqdm(train_loader)):
            step = epoch * len(train_loader) + batch_idx
            data = _move_to_device(data)
            score, label = model(**data)
            label = label[:][2].to(torch.float).cuda()
            loss = nn.BCELoss()(score, label)
            # loss = cosent(score, label)
            optimizer.zero_grad()
            loss.backward()
            step += 1
            optimizer.step()
            model.zero_grad()
            if step % args.report_step == 0:
                ndcg30 = evaluate(model, dev_loader, all_labels)
                logger.info('Epoch[{}/{}], loss:{}, ndcg30: {}'.format
                            (epoch + 1, args.num_epochs, loss.item(), ndcg30))
                model.train()
                if best < ndcg30:
                    early_stop = 0
                    best = ndcg30
                    torch.save(model.state_dict(), join(args.output_path, 'bert.pt'))
                    logger.info('higher_ndcg30: {}, step {}, epoch {}, save model\n'.format(best, step, epoch + 1))
                    continue
                early_stop += 1
                if early_stop == args.early_stop:
                    logger.info(f"ndcg30 doesn't improve for {early_stop} batch, early stop!")
                    logger.info(f"train use sample number: {(batch_idx - 10) * args.batch_size}")
                    logger.info('dev_best_ndcg30: {}'.format(best))
                    return


if __name__ == '__main__':
    for seed in range(1, 50):
        main(seed)
