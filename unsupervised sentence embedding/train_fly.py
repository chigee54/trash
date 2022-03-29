import os
import argparse
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pandas as pd
import numpy as np
from os.path import join
from loguru import logger
import einops
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from Custom_model import Custom_Model, simcse_unsup_loss, cosent_unsup_loss, SCD_Loss
from scipy.stats import spearmanr
from torch.utils.tensorboard import SummaryWriter


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
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


class PrepareDataset(Dataset):
    def __init__(self, args, tokenizer, eval_mode=None):
        self.overwrite_cache = args.overwrite_cache
        self.wiki_file = args.wiki_file
        self.nli_file = args.nli_file
        self.max_seq_length = args.max_seq_length
        self.tokenizer = tokenizer
        if eval_mode is not None:
            self.data = self.load_eval_data(eval_mode)
        else:
            self.data = self.load_wiki_unsupervised() if args.train_mode == 'unsupervised' else self.load_nli_supervised()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_wiki_unsupervised(self):
        logger.info('loading unsupervised train data')
        output_path = os.path.dirname('SimCSE-English/data')
        train_file_cache = join(output_path, 'train-unsupervise.pkl')
        if os.path.exists(train_file_cache) and not self.overwrite_cache:
            with open(train_file_cache, 'rb') as f:
                feature_list = pickle.load(f)
                logger.info("len of train data:{}".format(len(feature_list)))
                return feature_list
        feature_list = []
        with open(self.wiki_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            # lines = lines[:100]
            logger.info("len of train data:{}".format(len(lines)))
            for line in tqdm(lines):
                line = line.strip()
                feature = self.tokenizer([line, line], max_length=self.max_seq_length,
                                         truncation=True, padding='max_length', return_tensors='pt')
                feature_list.append(feature)
        with open(train_file_cache, 'wb') as f:
            pickle.dump(feature_list, f)
        return feature_list

    def load_train_data_supervised(self):
        logger.info('loading supervised train data')
        output_path = os.path.dirname('SimCSE-English/data')
        train_file_cache = join(output_path, 'train-supervised.pkl')
        if os.path.exists(train_file_cache) and not args.overwrite_cache:
            with open(train_file_cache, 'rb') as f:
                feature_list = pickle.load(f)
                logger.info("len of train data:{}".format(len(feature_list)))
                return feature_list
        feature_list = []
        df = pd.read_csv(self.nli_file, sep=',')
        logger.info("len of train data:{}".format(len(df)))
        rows = df.to_dict('records')
        # rows = rows[:10000]
        for row in tqdm(rows):
            sent0 = row['sent0']
            sent1 = row['sent1']
            hard_neg = row['hard_neg']
            feature = tokenizer([sent0, sent1, hard_neg], max_length=self.max_seq_length, truncation=True,
                                padding='max_length', return_tensors='pt')
            feature_list.append(feature)
        with open(train_file_cache, 'wb') as f:
            pickle.dump(feature_list, f)
        return feature_list

    def load_eval_data(self, mode):
        logger.info('loading {} data'.format(mode))
        output_path = os.path.dirname('output/unsupervise/')
        eval_file_cache = join(output_path, '{}_32.pkl'.format(mode))
        if os.path.exists(eval_file_cache) and not args.overwrite_cache:
            with open(eval_file_cache, 'rb') as f:
                feature_list = pickle.load(f)
                logger.info("len of {} data:{}".format(mode, len(feature_list)))
                return feature_list
        eval_file = args.dev_file if mode == 'dev' else args.test_file
        feature_list = []
        with open(eval_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            logger.info("len of {} data:{}".format(mode, len(lines)))
            for line in tqdm(lines):
                line = line.strip().split("\t")
                assert len(line) == 7 or len(line) == 9
                score = float(line[4])
                data1 = tokenizer(line[5].strip(), max_length=args.max_seq_length, truncation=True,
                                  padding='max_length', return_tensors='pt')
                data2 = tokenizer(line[6].strip(), max_length=args.max_seq_length, truncation=True,
                                  padding='max_length', return_tensors='pt')
                feature_list.append((data1, data2, score))
        with open(eval_file_cache, 'wb') as f:
            pickle.dump(feature_list, f)
        return feature_list


def da_function(y_pred, device, m, n):
    # batch=8时，8对正样本，16条句子，两两对比8*15=120对，故总共有240条句子/batch
    label, da_embedding = [], []
    for i in range(y_pred.shape[0]):
        for j in range(i + 1, i + m + 1):
            if j == y_pred.shape[0]:
                break
            if i % 3 == 0 and j == i + 1:
                label.append(2)
            elif i % 3 == 0 and j == i + 2 or i % 2 == 0 and j == i + 1:
                if str(label).count('1') > n:
                    continue
                label.append(1)
            else:
                label.append(0)
            if j == 1:
                da_embedding = torch.cat([y_pred[0].unsqueeze(0), y_pred[1].unsqueeze(0)], dim=0)
                continue
            temp = torch.cat([y_pred[i].unsqueeze(0), y_pred[j].unsqueeze(0)], dim=0)
            da_embedding = torch.cat([da_embedding, temp], dim=0)
    da_embedding = da_embedding.to(device)
    label = torch.tensor(label).to(device)
    return da_embedding, label


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


def evaluate(model, dataloader, cls_with_mlp=False):
    model.eval()
    sim_tensor = torch.tensor([], device=args.device)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(args.device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(args.device)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(args.device)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids, cls_with_mlp)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(args.device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(args.device)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(args.device)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids, cls_with_mlp)
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


def train(model, args):
    train_dataset = PrepareDataset(args, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataset = PrepareDataset(args, tokenizer, 'dev')
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)
    # 换成普通的优化器有时效果更好：optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    optimizer, scheduler = scheduler_with_optimizer(model, train_loader, args)
    best = 0
    for epoch in range(args.num_epochs):
        model.train()
        model.zero_grad()
        for batch_idx, data in enumerate(tqdm(train_loader)):
            step = epoch * len(train_loader) + batch_idx
            sql_len = data['input_ids'].shape[-1]
            input_ids = data['input_ids'].view(-1, sql_len).to(args.device)
            attention_mask = data['attention_mask'].view(-1, sql_len).to(args.device)
            token_type_ids = data['token_type_ids'].view(-1, sql_len).to(args.device)
            out = model(input_ids, attention_mask, token_type_ids)
            if args.loss == 'CL':
                loss = simcse_unsup_loss(out, args.device)
            elif args.loss == 'CS':
                out_list, labels = da_function(out, args.device, args.permutation_num, args.noise_num)
                loss = cosent_unsup_loss(out_list, labels)
            elif args.loss == 'SCD':
                out = einops.rearrange(out, '(h i) j -> h i j', h=2)
                loss = SCD_Loss(args)(out[0, :, :], out[1, :, :])
            else:
                loss = None
            # loss /= args.gradient_accumulation_step
            loss.backward()
            step += 1
            if step % args.gradient_accumulation_step == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            if step % args.report_step == 0:
                corrcoef = evaluate(model, dev_loader, args.do_eval_mlp)
                logger.info('Epoch[{}/{}], loss:{}, corrcoef: {}'.format
                            (epoch + 1, args.num_epochs, loss.item(), corrcoef))
                writer.add_scalar('loss', loss, step)
                writer.add_scalar('corrcoef', corrcoef, step)
                model.train()
                if best < corrcoef:
                    best = corrcoef
                    torch.save(model.state_dict(), join(args.output_path, 'bert.pt'))
                    logger.info('higher_corrcoef: {}, step {}, epoch {}, save model\n'.format(best, step, epoch + 1))
    logger.info('dev_best_corrcoef: {}'.format(best))


def test(model, args):
    test_dataset = PrepareDataset(args, tokenizer, 'test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    model.load_state_dict(torch.load(join(args.output_path, 'bert.pt')))
    corrcoef = evaluate(model, test_loader, args.do_eval_mlp)
    logger.info('test corrcoef:{}'.format(corrcoef))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='cuda', type=str)
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-pre_seq_len', default=16, type=int)
    parser.add_argument('-dropout', default=0.1, type=float)
    parser.add_argument('-dropout_noise', default=0.5, type=float)
    parser.add_argument('-permutation_num', default=4, type=float)
    parser.add_argument('-noise_num', default=32, type=float)
    parser.add_argument('-alpha', default=1.0, type=float)
    parser.add_argument('-beta', default=0.005225, type=float)
    parser.add_argument('-lambd', default=0.012, type=float)
    parser.add_argument('-temp', default=0.05, type=float)
    parser.add_argument('-embedding_dim', default=768, type=int)
    parser.add_argument('-projector_num', default="4096-4096-4096", type=str)
    parser.add_argument('-max_seq_length', default=32, type=int)
    parser.add_argument('-batch_size', default=256, type=int)
    parser.add_argument('-eval_batch_size', default=256, type=int)
    parser.add_argument('-num_epochs', default=1, type=int)
    parser.add_argument('-learning_rate', default=3e-2, type=float)
    parser.add_argument('-max_grad_norm', default=1.0, type=float)
    parser.add_argument('-warm_up_proportion', default=0, type=float)
    parser.add_argument('-gradient_accumulation_step', default=1, type=int)
    parser.add_argument('-report_step', default=125, type=int)
    parser.add_argument('-do_train', default=True, type=bool)
    parser.add_argument('-do_test', default=True, type=bool)
    parser.add_argument('-do_ptuning', default=True, type=bool)
    parser.add_argument('-do_eval_mlp', default=False, type=bool)
    parser.add_argument('-multi_dropout', default=False, type=bool)
    parser.add_argument('-overwrite_cache', default=False, type=bool)
    parser.add_argument('-pool_type', default='cls', type=str)
    parser.add_argument('-loss', default='CL', type=str)
    parser.add_argument('-train_mode', default='unsupervised', type=str)
    parser.add_argument('-output_path', default='output/try3', type=str)
    parser.add_argument('-bert_path', default='pretrained_model/bert_based_uncased_english', type=str)
    parser.add_argument('-wiki_file', default='SimCSE-English/data/wiki1m_for_simcse.txt', type=str)
    parser.add_argument('-nli_file', default='SimCSE-English/data/nli_for_simcse.csv', type=str)
    parser.add_argument('-train_file', default='SentEval/data/downstream/STS/STSBenchmark/sts-train.csv', type=str)
    parser.add_argument('-dev_file', default='SentEval/data/downstream/STS/STSBenchmark/sts-dev.csv', type=str)
    parser.add_argument('-test_file', default='SentEval/data/downstream/STS/STSBenchmark/sts-test.csv', type=str)
    args = parser.parse_args()

    # config environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)
    args.output_path = join(args.output_path, args.train_mode, 'bsz-{}-lr-{}-dropout-{}'.
                              format(args.batch_size, args.learning_rate, args.dropout))
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # model initial
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    model = Custom_Model(args.bert_path, pooling=args.pool_type, pre_seq_len=args.pre_seq_len,
                         multi_dropout=args.multi_dropout, dropout_noise=args.dropout_noise,
                         dropout=args.dropout, do_ptuning=args.do_ptuning)
    model = torch.nn.DataParallel(model)
    model.to(args.device)

    # save log and tensorboard
    cur_time = time.strftime("%Y%m%d_%H_%M", time.localtime())
    logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
    logger.info(args)
    writer = SummaryWriter(args.output_path)

    # run
    start_time = time.time()
    if args.do_train:
        train(model, args)
    if args.do_test:
        test(model, args)
    logger.info("run time: {:.4f}".format((time.time() - start_time)/60))
