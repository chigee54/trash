import pickle

from tqdm import tqdm
import json
import os
import time
import math
from loguru import logger
import numpy as np
from os.path import join
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import random
import argparse
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
# from modeling_longformer import LongformerModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='cuda', type=str)
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-label_file', default='./data/phase_1/train/label_top30_dict.json', type=str)
    parser.add_argument('-train_file', default='data/phase_2/train_data_win200_plus.json', type=str)
    parser.add_argument('-dev_file', default='data/phase_2/dev_data_win200_plus.json', type=str)
    parser.add_argument('-model_path', default='../pretrained_model/lawformer', type=str)
    parser.add_argument('-output_path', default='./result', type=str)
    parser.add_argument('-report_step', default=1000, type=int)
    parser.add_argument('-early_stop', default=1000, type=int)
    parser.add_argument('-max_length', default=1533, type=int)
    parser.add_argument('-eval_batch_size', default=20, type=int)
    parser.add_argument('-batch_size', default=2, type=int)
    parser.add_argument('-num_epochs', default=10, type=int)
    parser.add_argument('-learning_rate', default=8e-6, type=float)
    parser.add_argument('-warm_up_proportion', default=0, type=float)
    parser.add_argument('-gradient_accumulation_step', default=1, type=int)
    args = parser.parse_args()

    # config environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)
    args.output_path = join(args.output_path, 'bsz-{}-lr-{}'.format(args.batch_size, args.learning_rate))
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    cur_time = time.strftime("%Y%m%d_%H_%M", time.localtime())
    logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
    logger.info(args)

    tokenizer = AutoTokenizer.from_pretrained("../pretrained_model/roberta")
    tokenizer.add_tokens(['☢'])
    model = LongModel(args.model_path, tokenizer).cuda()
    # model = nn.DataParallel(model, device_ids=[0, 1], output_device=0)
    start_time = time.time()
    train(model, tokenizer, args)
    logger.info('run time: {:.2f}'.format((time.time() - start_time) / 60))


def seed_everything(seed):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


class LongModel(nn.Module):
    def __init__(self, model_path, tokenizer):
        super(LongModel, self).__init__()
        # len_reduce_list = [int(1280 * (0.90) ** i) for i in range(1, 13)]
        # len_reduce_list = [1536, 1536, 1536, 1536, 1024, 1024, 1024, 1024, 512, 512, 512, 512]
        # gate_type = 'pagerank'
        self.model = AutoModel.from_pretrained(model_path,
                                                     # output_attentions=True,
                                                     # len_reduce_list=len_reduce_list,
                                                     # gate_type=gate_type
                                                     )
        self.model.resize_token_embeddings(len(tokenizer))
        self.linear = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, label=None):
        output = self.model(input_ids, attention_mask, token_type_ids)

        logits = self.linear(output[1])
        # score = torch.sigmoid(logits).squeeze(-1)
        return logits, label


class LongDataset(Dataset):
    def __init__(self, data_file, tokenizer):
        self.tokenizer = tokenizer
        with open(data_file, 'r', encoding='utf8') as f:
            self.data = json.load(f)
            print(len(self.data))

    def __getitem__(self, index):
        data = self.data[index]
        # crime_token_id = self.tokenizer.convert_tokens_to_ids(['☢'])
        query_cut = self.tokenizer.tokenize(data['crime']+'☢'+data['query'])[:509]
        candidate_cut = self.tokenizer.tokenize(data['cpfxgc']+'☢'+data['ajjbqk'])[:1020]
        tokens = ['[CLS]'] + query_cut + ['[SEP]'] + candidate_cut + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        types = [0] * (len(query_cut) + 2) + [1] * (len(candidate_cut) + 1)
        # s1_sep, s2_sep = np.where(np.array(token_ids) == self.tokenizer.sep_token_id)[0]
        # s1_title, s2_title = np.where(np.array(token_ids) == crime_token_id)[0]
        # gate_mask = np.zeros(len(token_ids), dtype=np.int32)
        # gate_mask[:s1_title + 1] = 1
        # gate_mask[s1_sep:s2_title + 1] = 1
        # gate_mask[s2_sep] = 1
        # gate_mask = gate_mask.tolist()
        data['labels'][2] /= 3
        input_ids, token_type_ids, attention_mask = self.pad_seq(token_ids, types)
        feature = {'input_ids': torch.LongTensor(input_ids),
                   'token_type_ids': torch.LongTensor(token_type_ids),
                   'attention_mask': torch.LongTensor(attention_mask),
                   # 'gate_mask': torch.LongTensor(gate_mask),
                   'label': data['labels']
                   }
        return feature

    def pad_seq(self, ids, types):
        batch_len = 1533
        masks = [1] * len(ids) + [0] * (batch_len - len(ids))
        types += [0] * (batch_len - len(ids))
        # gate += [0] * (batch_len - len(ids))
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
        ndcgs.append(ndcg(ranks, 30))
        # print(f'********** qidx: {qidx} **********')
        # print(f'top30 pred_ids: {pred_ids}')
        # print(f'ranks: {ranks}')
    # print(ndcgs)
    return sum(ndcgs) / len(ndcgs)


class FocalLoss(nn.Module):

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def evaluate(model, dataloader, all_labels):
    model.eval()
    all_labels, sub_labels = {}, {}
    with torch.no_grad():
        all_preds, info = {}, {}
        for data in dataloader:
            data = _move_to_device(data)
            score, label = model(**data)
            # score = list(torch.max(score, 1)[1].cpu().numpy())
            for n, i in enumerate(zip(label[0], label[1], label[2])):
                if i[0] not in info.keys():
                    sub_labels[i[1]] = i[2]
                    all_labels[i[0]] = sub_labels
                    info[i[0]] = [[i[1]], [score[n]]]
                else:
                    all_labels[i[0]][i[1]] = i[2]
                    info[i[0]][1].append(score[n])
                    info[i[0]][0].append(i[1])
        for qidx in info.keys():
            dids, preds = info[qidx]
            sorted_r = sorted(list(zip(dids, preds)), key=lambda x: x[1], reverse=True)
            pred_ids = [x[0] for x in sorted_r]
            all_preds[qidx] = pred_ids[:30]
    ndcg_30 = cal_ndcg(all_preds, all_labels)
    del info, all_preds
    torch.cuda.empty_cache()
    return ndcg_30


def train(model, tokenizer, args):
    all_labels = json.load(open(args.label_file, 'r', encoding='utf8'))
    train_data = LongDataset(args.train_file, tokenizer)
    dev_data = LongDataset(args.dev_file, tokenizer)
    dev_loader = DataLoader(dev_data, batch_size=args.eval_batch_size, shuffle=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    optimizer, scheduler = scheduler_with_optimizer(model, train_loader, args)
    loss_function = nn.MSELoss()
    # loss_function = FocalLoss(class_num=4)
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
            label = label[:][2]
            loss = loss_function(score.view(-1), label.to(torch.float).cuda())
            optimizer.zero_grad()
            loss.backward()
            step += 1
            optimizer.step()
            model.zero_grad()
            if step % args.report_step == 0:
                torch.cuda.empty_cache()
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
    main()
