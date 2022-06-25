import os
import argparse
import time
import pickle
import torch
from evaluation_ease import do_senteval
import torch.nn.functional as F
import json
import random
import pandas as pd
import numpy as np
from os.path import join
from loguru import logger
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AdamW, BertForPreTraining, get_linear_schedule_with_warmup, AutoConfig, BertTokenizer
from models import BertForCL, RobertaForCL
from scipy.stats import spearmanr
from torch.utils.tensorboard import SummaryWriter
import random


def main(seed=17, alpha=0.11):
    print('\nAlpha: {}\n'.format(alpha))
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='cuda', type=str)
    parser.add_argument('-seed', default=seed, type=int)
    parser.add_argument('-alpha', default=alpha, type=float)
    parser.add_argument('-dropout', default=0.1, type=float)
    parser.add_argument('-dropout_noise', default=0.83, type=float)
    parser.add_argument('-temp', default=0.05, type=float)
    parser.add_argument('-max_seq_length', default=32, type=int)
    parser.add_argument('-batch_size', default=64, type=int)
    parser.add_argument('-eval_batch_size', default=256, type=int)
    parser.add_argument('-num_epochs', default=1, type=int)
    parser.add_argument('-learning_rate', default=2e-5, type=float)
    parser.add_argument('-max_grad_norm', default=1.0, type=float)
    parser.add_argument('-warm_up_proportion', default=0, type=float)
    parser.add_argument('-gradient_accumulation_step', default=1, type=int)
    parser.add_argument('-report_step', default=125, type=int)
    parser.add_argument('-do_train', default=True, type=bool)
    parser.add_argument('-do_test', default=True, type=bool)
    parser.add_argument('-do_senteval', default=True, type=bool)
    parser.add_argument('-do_ptuning', default=False, type=bool)
    parser.add_argument('-mlp_only_train', default=True, type=bool)
    parser.add_argument('-before_mlp', default=True, type=bool)
    parser.add_argument('-batchnorm', default=False, type=bool)
    parser.add_argument('-multi_dropout', default=True, type=bool)
    parser.add_argument('-overwrite_cache', default=False, type=bool)
    parser.add_argument('-train_bert', default=False, type=bool)
    parser.add_argument('-demo_train', default=False, type=bool)
    parser.add_argument('-model_type', default='bert', type=str)
    parser.add_argument('-pooler_type', default='cls', type=str)
    parser.add_argument('-train_mode', default='unsupervised', choices=['supervised', 'supervised_sts', 'unsupervised'], type=str)
    parser.add_argument('-output_path', default='output/try7/{}'.format(seed), type=str)
    parser.add_argument('-roberta_path', default=r'D:\zzj_project\PI_project\Law\pretrained_model\roberta_base', type=str)
    parser.add_argument('-bert_path', default='../Law/pretrained_model/bert_based_uncased_english', type=str)
    parser.add_argument('-wiki_file', default='../Law/SimCSE-English/data/wiki1m_for_simcse.txt', type=str)
    parser.add_argument('-nli_file', default='../Law/SimCSE-English/data/nli_for_simcse.csv', type=str)
    parser.add_argument('-train_file', default='../Law/SentEval/data/downstream/STS/STSBenchmark/sts-train.csv', type=str)
    parser.add_argument('-dev_file', default='../Law/SentEval/data/downstream/STS/STSBenchmark/sts-dev.csv', type=str)
    parser.add_argument('-test_file', default='../Law/SentEval/data/downstream/STS/STSBenchmark/sts-test.csv', type=str)
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
    config = AutoConfig.from_pretrained(args.bert_path)
    if args.multi_dropout:
        config.multi_dropout = args.multi_dropout
        config.attention_probs_dropout_prob_noise = args.dropout_noise
        config.hidden_dropout_prob_noise = args.dropout_noise
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    if args.model_type == 'bert':
        model = BertForCL.from_pretrained(args.bert_path, config=config, args=args)
    else:
        model = RobertaForCL.from_pretrained(args.bert_path, config=config, args=args)
    # pretrained_model = BertForPreTraining.from_pretrained(args.bert_path)
    # model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
    model.to(args.device)

    # save log and tensorboard
    cur_time = time.strftime("%Y%m%d_%H_%M", time.localtime())
    logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
    logger.info(args)
    writer = SummaryWriter(args.output_path)

    # run
    start_time = time.time()
    if args.do_train:
        train(model, tokenizer, writer, args)
    if args.do_test:
        test(model, tokenizer, args)
    if args.do_senteval:
        SentEval(model, tokenizer, args)
    logger.info("run time: {:.2f}".format((time.time() - start_time) / 60))


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


class PrepareDataset(Dataset):
    def __init__(self, args, tokenizer, eval_mode=None):
        self.args = args
        self.overwrite_cache = args.overwrite_cache
        self.wiki_file = args.wiki_file
        self.nli_file = args.nli_file
        self.train_file = args.train_file
        self.dev_file = args.dev_file
        self.test_file = args.test_file
        self.max_seq_length = args.max_seq_length
        self.tokenizer = tokenizer
        if eval_mode is not None:
            self.data = self.load_eval_data(eval_mode)
        else:
            if args.train_mode == 'unsupervised':
                self.data = self.load_wiki_unsupervised()
            else:
                self.data = self.load_sts_supervised() if args.train_mode == 'supervised_sts' else self.load_nli_supervised()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_wiki_unsupervised(self):
        logger.info('loading unsupervised train data')
        output_path = os.path.dirname('../Law/SimCSE-English/data')
        # if self.args.multi_dropout is True:
        #     train_file_cache = join(output_path, 'train-unsupervise_3.pkl')if self.args.demo_train is False else join(output_path, 'demo1_train.pkl')
        # else:
        train_file_cache = join(output_path, 'train-unsupervise.pkl') if self.args.demo_train is False else join(output_path, 'demo_2_train.pkl')
        if os.path.exists(train_file_cache) and not self.overwrite_cache:
            with open(train_file_cache, 'rb') as f:
                feature_list = pickle.load(f)
                logger.info("len of train data:{}".format(len(feature_list)))
                return feature_list
        feature_list = []
        with open(self.wiki_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            # lines = lines[:50000]
            logger.info("len of train data:{}".format(len(lines)))
            for line in tqdm(lines):
                line = line.strip()
                feature = self.tokenizer([line, line], max_length=self.max_seq_length,
                                         truncation=True, padding='max_length', return_tensors='pt')
                feature_list.append(feature)
        with open(train_file_cache, 'wb') as f:
            pickle.dump(feature_list, f)
        return feature_list

    def load_nli_supervised(self):
        logger.info('loading supervised train data')
        output_path = os.path.dirname('../Law/SimCSE-English/data')
        train_file_cache = join(output_path, 'train-supervised.pkl')
        if os.path.exists(train_file_cache) and not self.overwrite_cache:
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
            feature = self.tokenizer([sent0, sent1, hard_neg], max_length=self.max_seq_length, truncation=True,
                                padding='max_length', return_tensors='pt')
            feature_list.append(feature)
        with open(train_file_cache, 'wb') as f:
            pickle.dump(feature_list, f)
        return feature_list

    def load_sts_supervised(self):
        logger.info('loading supervised train data')
        output_path = os.path.dirname('../Law/SimCSE-English/data')
        train_file_cache = join(output_path, 'train-supervised_stsb.pkl')
        if os.path.exists(train_file_cache) and not self.overwrite_cache:
            with open(train_file_cache, 'rb') as f:
                feature_list = pickle.load(f)
                logger.info("len of train data:{}".format(len(feature_list)))
                return feature_list
        feature_list = []
        with open(self.train_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            logger.info("len of data:{}".format(len(lines)))
            for line in tqdm(lines):
                line = line.strip().split("\t")
                assert len(line) == 7 or len(line) == 9
                score = float(line[4])
                feature = self.tokenizer([line[5], line[6]], max_length=self.max_seq_length, truncation=True,
                                         padding='max_length', return_tensors='pt')
                feature_list.append((feature, score))
        with open(train_file_cache, 'wb') as f:
            pickle.dump(feature_list, f)
        return feature_list

    def load_eval_data(self, mode):
        logger.info('loading {} data'.format(mode))
        output_path = os.path.dirname('../Law/output/unsupervise/')
        eval_file_cache = join(output_path, '{}_32.pkl'.format(mode))
        if os.path.exists(eval_file_cache) and not self.overwrite_cache:
            with open(eval_file_cache, 'rb') as f:
                feature_list = pickle.load(f)
                logger.info("len of {} data:{}".format(mode, len(feature_list)))
                return feature_list
        eval_file = self.dev_file if mode == 'dev' else self.test_file
        feature_list = []
        with open(eval_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            logger.info("len of {} data:{}".format(mode, len(lines)))
            for line in tqdm(lines):
                line = line.strip().split("\t")
                assert len(line) == 7 or len(line) == 9
                score = float(line[4])
                data1 = self.tokenizer(line[5].strip(), max_length=self.max_seq_length, truncation=True,
                                  padding='max_length', return_tensors='pt')
                data2 = self.tokenizer(line[6].strip(), max_length=self.max_seq_length, truncation=True,
                                  padding='max_length', return_tensors='pt')
                feature_list.append((data1, data2, score))
        with open(eval_file_cache, 'wb') as f:
            pickle.dump(feature_list, f)
        return feature_list


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


def evaluate(model, dataloader, args):
    model.eval()
    sim_tensor = torch.tensor([], device=args.device)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            source_token_type_ids, target_token_type_ids = None, None
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(args.device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(args.device)
            if source.get('token_type_ids') is not None:
                source_token_type_ids = source.get('token_type_ids').squeeze(1).to(args.device)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)[1]
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(args.device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(args.device)
            if target.get('token_type_ids') is not None:   
                target_token_type_ids = target.get('token_type_ids').squeeze(1).to(args.device)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)[1]
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


def train(model, tokenizer, writer, args):
    train_dataset = PrepareDataset(args, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataset = PrepareDataset(args, tokenizer, 'dev')
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)
    optimizer, scheduler = scheduler_with_optimizer(model, train_loader, args)
    best = 0
    for epoch in range(args.num_epochs):
        model.train()
        model.zero_grad()
        early_stop = 0
        for batch_idx, data in enumerate(tqdm(train_loader)):
            step = epoch * len(train_loader) + batch_idx
            input_ids = data['input_ids'].cuda()
            attention_mask = data['attention_mask'].cuda()
            token_type_ids = None
            if data['token_type_ids'] is not None:
                token_type_ids = data['token_type_ids'].cuda()
            if args.multi_dropout:
                input_ids = torch.cat([input_ids, input_ids[:, 0, :].unsqueeze(1)], dim=1)
                attention_mask = torch.cat([attention_mask, attention_mask[:, 0, :].unsqueeze(1)], dim=1)
                if token_type_ids is not None:
                    token_type_ids = torch.cat([token_type_ids, token_type_ids[:, 0, :].unsqueeze(1)], dim=1)
            data = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'sent_emb': False
            }
            out = model(**data)
            loss = out[0]
            # loss /= args.gradient_accumulation_step

            optimizer.zero_grad()
            loss.backward()
            step += 1
            # if step % args.gradient_accumulation_step == 0:
            #     nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            # scheduler.step()
            model.zero_grad()
            if step % args.report_step == 0:
                corrcoef = evaluate(model, dev_loader, args)
                logger.info('Epoch[{}/{}], loss:{}, corrcoef: {}'.format
                            (epoch + 1, args.num_epochs, loss.item(), corrcoef))
                writer.add_scalar('loss', loss, step)
                writer.add_scalar('corrcoef', corrcoef, step)
                model.train()
                if best < corrcoef:
                    early_stop = 0
                    best = corrcoef
                    torch.save(model.state_dict(), join(args.output_path, 'bert.pt'))
                    logger.info('higher_corrcoef: {}, step {}, epoch {}, save model\n'.format(best, step, epoch + 1))
                    continue
                early_stop += 1
                if early_stop == 50:
                    logger.info(f"corrcoef doesn't improve for {early_stop} batch, early stop!")
                    logger.info(f"train use sample number: {(batch_idx - 10) * args.batch_size}")
                    logger.info('dev_best_corrcoef: {}'.format(best))
                    return


def test(model, tokenizer, args):
    test_dataset = PrepareDataset(args, tokenizer, 'test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    model.load_state_dict(torch.load(join(args.output_path, 'bert.pt')))
    corrcoef = evaluate(model, test_loader, args)
    logger.info('test corrcoef:{}'.format(corrcoef))


def SentEval(model, tokenizer, args):
    model.load_state_dict(torch.load(join(args.output_path, 'bert.pt')))
    result_STS, result_transfer = do_senteval(tokenizer, model, args.device)
    logger.info('\n{}'.format(result_STS))


if __name__ == '__main__':
    for seed in range(17, 100):
        main(seed=seed)
