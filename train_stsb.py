import os
import argparse
import time
import torch
import torch.nn as nn
import random
import numpy as np
from os.path import join
from loguru import logger
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
# from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
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


def load_data(path):
    input_file = open(path, encoding='utf-8')
    lines = input_file.readlines()[1:]
    input_file.close()
    input_ids, attention_mask, token_type_ids = [], [], []
    labels = []
    print('load data')
    for line in tqdm(lines):
        line_split = line.strip().split("\t")
        assert len(line_split) == 7
        ans = tokenizer.encode_plus(line_split[5], line_split[6], max_length=args.max_seq_length,
                                    padding="max_length", truncation="longest_first")
        input_ids.append(ans.input_ids)
        attention_mask.append(ans.attention_mask)
        token_type_ids.append(ans.token_type_ids)
        labels.append(float(line_split[4]))
    return np.array(input_ids), np.array(attention_mask), np.array(token_type_ids), np.array(labels)


def data_loader(data_file, batch_size, shuffle=True):
    input_ids, attention_mask, token_type_ids, labels = load_data(data_file)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float)
    data = TensorDataset(input_ids, attention_mask, token_type_ids, labels)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader


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


def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
        preds = []
        label_array = np.array([])
        for cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_y in dataloader:
            cur_input_ids = cur_input_ids.to(args.device)
            cur_attention_mask = cur_attention_mask.to(args.device)
            cur_token_type_ids = cur_token_type_ids.to(args.device)
            outputs = model(cur_input_ids, cur_attention_mask, cur_token_type_ids)
            label_array = np.append(label_array, np.array(cur_y))
            preds.extend(list(outputs[0].view(-1).cpu().numpy()))
        preds = np.clip(np.array(preds), 0, 5)
        cur_pearsonr = pearsonr(label_array, preds)[0]
        cur_spearmanr = spearmanr(label_array, preds)[0]
    return cur_pearsonr, cur_spearmanr


def train(model, args):
    train_loader = data_loader(args.train_file, args.batch_size)
    dev_loader = data_loader(args.dev_file, args.eval_batch_size)
    optimizer, scheduler = scheduler_with_optimizer(model, train_loader, args)
    best = 0
    for epoch in range(args.num_epochs):
        model.train()
        model.zero_grad()
        for batch_idx, (cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_y) in enumerate(tqdm(train_loader)):
            step = epoch * len(train_loader) + batch_idx
            cur_input_ids = cur_input_ids.to(args.device)
            cur_attention_mask = cur_attention_mask.to(args.device)
            cur_token_type_ids = cur_token_type_ids.to(args.device)
            cur_y = cur_y.to(args.device)
            outputs = model(cur_input_ids, cur_attention_mask, cur_token_type_ids)
            loss = nn.MSELoss()(outputs[0].view(-1), cur_y)
            loss /= args.gradient_accumulation_step
            loss.backward()
            step += 1
            if step % args.gradient_accumulation_step == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            if step % args.report_step == 0:
                _, corrcoef = evaluate(model, dev_loader)
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
    test_loader = data_loader(args.test_file, args.eval_batch_size)
    model.load_state_dict(torch.load(join(args.output_path, 'bert.pt')))
    _, corrcoef = evaluate(model, test_loader)
    logger.info('test corrcoef:{}'.format(corrcoef))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='cuda', type=str)
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-max_seq_length', default=128, type=int)
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('-eval_batch_size', default=256, type=int)
    parser.add_argument('-num_epochs', default=3, type=int)
    parser.add_argument('-learning_rate', default=2e-5, type=float)
    parser.add_argument('-max_grad_norm', default=1.0, type=float)
    parser.add_argument('-warm_up_proportion', default=0.1, type=float)
    parser.add_argument('-gradient_accumulation_step', default=1, type=int)
    parser.add_argument('-bert_path', default='pretrained_model/bert_based_uncased_english', type=str)
    parser.add_argument('-dataset', default='STS-B', type=str)
    parser.add_argument('-report_step', default=10, type=int)
    parser.add_argument('-output_path', default='glue_data/output', type=str)
    parser.add_argument('-do_train', default=True, type=bool)
    parser.add_argument('-do_test', default=True, type=bool)
    parser.add_argument('-train_file', default='SentEval/data/downstream/STS/STSBenchmark/sts-train.csv', type=str)
    parser.add_argument('-dev_file', default='SentEval/data/downstream/STS/STSBenchmark/sts-dev.csv', type=str)
    parser.add_argument('-test_file', default='SentEval/data/downstream/STS/STSBenchmark/sts-test.csv', type=str)
    args = parser.parse_args()

    # config environment
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # model initial
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    model = BertForSequenceClassification.from_pretrained(args.bert_path, num_labels=1)
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
