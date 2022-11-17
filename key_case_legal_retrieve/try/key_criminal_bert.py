import pickle

from tqdm import tqdm
import json
import os
import time
import torch.nn.functional as F
from torch.autograd import Variable
from loguru import logger
import numpy as np
from os.path import join
import torch
import random
import argparse
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='cuda', type=str)
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-label_file', default='./data/phase_1/train/label_top30_dict.json', type=str)
    parser.add_argument('-train_file', default='data/phase_2/train_data.json', type=str)
    parser.add_argument('-dev_file', default='data/phase_2/dev_data.json', type=str)
    parser.add_argument('-model_path', default='../pretrained_model/criminal_bert', type=str)
    parser.add_argument('-output_path', default='./result/case', type=str)
    parser.add_argument('-report_step', default=100, type=int)
    parser.add_argument('-early_stop', default=1000, type=int)
    parser.add_argument('-max_length', default=1280, type=int)
    parser.add_argument('-eval_batch_size', default=24, type=int)
    parser.add_argument('-batch_size', default=8, type=int)
    parser.add_argument('-num_epochs', default=10, type=int)
    parser.add_argument('-learning_rate', default=1e-5, type=float)
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

    model = LongModel(args.model_path).cuda()
    # model = nn.DataParallel(model, device_ids=[0, 1], output_device=0)
    start_time = time.time()
    train(model, args)
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
    def __init__(self, model_path):
        super(LongModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.linear = nn.Linear(self.model.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids, label=None):
        output = self.model(input_ids, attention_mask, token_type_ids)
        pred = self.linear(output[1])
        # pred = torch.sigmoid(pred).squeeze(-1)
        return pred, label


class LongDataset(Dataset):
    def __init__(self, data_file):
        self.tokenizer = AutoTokenizer.from_pretrained("../pretrained_model/criminal_bert")
        with open(data_file, 'r', encoding='utf8') as f:
            all_data = json.load(f)
            data = []
            for i in range(len(all_data)//100):
                data_part = all_data[i*100:100*(i+1)]
                top_30_data = sorted(data_part, key=lambda x: x['labels'][2], reverse=True)[:30]
                data.extend(top_30_data)
            self.data = data
            print(len(self.data))

    def __getitem__(self, index):
        data = self.data[index]
        query_cut = self.tokenizer.tokenize(data['crime'])
        candidate_cut = self.tokenizer.tokenize(data['ajName'])
        tokens = ['[CLS]'] + query_cut + ['[SEP]'] + candidate_cut + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        types = [0] * (len(query_cut) + 2) + [1] * (len(candidate_cut) + 1)
        input_ids, token_type_ids, attention_mask = self.pad_seq(token_ids, types)
        try:
            if data['labels'][2] == 1/3 or data['labels'][2] == 0.0:
                data['labels'][2] = 0
            else:
                data['labels'][2] = 1
        except:
            exit(data['labels'])
        feature = {'input_ids': torch.LongTensor(input_ids),
                   'token_type_ids': torch.LongTensor(token_type_ids),
                   'attention_mask': torch.LongTensor(attention_mask),
                   'label': data['labels']
                   }
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


def _move_to_device(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()
    return batch


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


def evaluate(model, dataloader):
    model.eval()
    all_score, all_label = [], []
    with torch.no_grad():
        for data in dataloader:
            data = _move_to_device(data)
            score, label = model(**data)
            all_label.extend(label[2])
            # score = (score > 0.5).type(torch.long)
            # score = score.cpu().detach().tolist()
            score = list(torch.max(score, 1)[1].cpu().numpy())
            all_score.extend(score)
    accuracy = accuracy_score(all_label, all_score)
    return accuracy


def train(model, args):
    train_data = LongDataset(args.train_file)
    dev_data = LongDataset(args.dev_file)
    dev_loader = DataLoader(dev_data, batch_size=args.eval_batch_size, shuffle=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    optimizer, scheduler = scheduler_with_optimizer(model, train_loader, args)
    loss_function = FocalLoss(2)
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
            label = label[:][2].to(torch.long).cuda()
            loss = loss_function(score.view(-1, 2), label)
            optimizer.zero_grad()
            loss.backward()
            step += 1
            optimizer.step()
            model.zero_grad()
            if step % args.report_step == 0:
                torch.cuda.empty_cache()
                acc = evaluate(model, dev_loader)
                logger.info('Epoch[{}/{}], loss:{}, acc: {}'.format
                            (epoch + 1, args.num_epochs, loss.item(), acc))
                model.train()
                if best < acc:
                    early_stop = 0
                    best = acc
                    torch.save(model.state_dict(), join(args.output_path, 'bert.pt'))
                    logger.info('higher_acc: {}, step {}, epoch {}, save model\n'.format(best, step, epoch + 1))
                    continue
                early_stop += 1
                if early_stop == args.early_stop:
                    logger.info(f"acc doesn't improve for {early_stop} batch, early stop!")
                    logger.info(f"train use sample number: {(batch_idx - 10) * args.batch_size}")
                    logger.info('dev_best_acc: {}'.format(best))
                    return


if __name__ == '__main__':
    main()
