import os
import argparse
import time
import pickle
import torch
from evaluation_ease import do_senteval
import torch.nn.functional as F
import random
import pandas as pd
import numpy as np
from os.path import join
from loguru import logger
import einops
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertForMaskedLM
from 2_custom_model import Custom_Model, simcse_unsup_loss, cosent_loss, SCD_Loss, simcse_sup_loss, SNCSE_loss, prompt_loss
# from Custom_model import Custom_Model, simcse_unsup_loss, cosent_loss, SCD_Loss, simcse_sup_loss
from scipy.stats import spearmanr
from torch.utils.tensorboard import SummaryWriter


def main(seed):
    print('\nSEED: {}\n'.format(seed))
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='cuda', type=str)
    parser.add_argument('-seed', default=seed, type=int)
    parser.add_argument('-pre_seq_len', default=16, type=int)
    parser.add_argument('-dropout', default=0.1, type=float)
    parser.add_argument('-dropout_noise', default=0.1, type=float)
    parser.add_argument('-permute_num', default=16, type=float)
    parser.add_argument('-noise_num', default=16, type=float)
    parser.add_argument('-alpha', default=1.0, type=float)
    parser.add_argument('-beta', default=0.005225, type=float)
    parser.add_argument('-lambd', default=0.012, type=float)
    parser.add_argument('-temp', default=0.05, type=float)
    parser.add_argument('-embedding_dim', default=768, type=int)
    parser.add_argument('-projector_num', default="4096-4096-4096", type=str)
    parser.add_argument('-max_seq_length', default=32, type=int)
    parser.add_argument('-batch_size', default=64, type=int)
    parser.add_argument('-eval_batch_size', default=256, type=int)
    parser.add_argument('-num_epochs', default=1, type=int)
    parser.add_argument('-learning_rate', default=3e-5, type=float)
    parser.add_argument('-max_grad_norm', default=1.0, type=float)
    parser.add_argument('-warm_up_proportion', default=0, type=float)
    parser.add_argument('-gradient_accumulation_step', default=1, type=int)
    parser.add_argument('-report_step', default=125, type=int)
    parser.add_argument('-do_train', default=True, type=bool)
    parser.add_argument('-do_test', default=True, type=bool)
    parser.add_argument('-do_senteval', default=True, type=bool)
    parser.add_argument('-do_ptuning', default=False, type=bool)
    parser.add_argument('-mlp_cls', default=False, type=bool)
    parser.add_argument('-multi_dropout', default=False, type=bool)
    parser.add_argument('-overwrite_cache', default=False, type=bool)
    parser.add_argument('-train_bert', default=False, type=bool)
    parser.add_argument('-demo_train', default=False, type=bool)
    parser.add_argument('-adv', default=True, type=bool)
    parser.add_argument('-pool_type', default='cls', type=str)
    parser.add_argument('-loss', default='CL', choices=['CL', 'CS', 'SCD', 'SUP_NLI', 'SUP_STS', 'SNCSE', 'PL'], type=str)
    parser.add_argument('-train_mode', default='unsupervised', choices=['supervised', 'supervised_sts', 'unsupervised'], type=str)
    parser.add_argument('-output_path', default='output/try8/{}'.format(seed), type=str)
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
    # model = Custom_Model(args.bert_path, pooling=args.pool_type, pre_seq_len=args.pre_seq_len,
    #                      multi_dropout=args.multi_dropout, dropout_noise=args.dropout_noise,
    #                      dropout=args.dropout, do_ptuning=args.do_ptuning, train_bert=args.train_bert)
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
        output_path = os.path.dirname('SimCSE-English/data')
        if self.args.multi_dropout is True:
            train_file_cache = join(output_path, 'train-unsupervise_3.pkl')if self.args.demo_train is False else join(output_path, 'demo1_train.pkl')
        else:
            train_file_cache = join(output_path, 'train-unsupervise.pkl') if self.args.demo_train is False else join(output_path, 'demo_2_train.pkl')
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

    def load_nli_supervised(self):
        logger.info('loading supervised train data')
        output_path = os.path.dirname('SimCSE-English/data')
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
        output_path = os.path.dirname('SimCSE-English/data')
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
        output_path = os.path.dirname('output/unsupervise/')
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


def da_function(y_pred, device, permute_num, noise_num, triple=True):
    # batch=8时，8对正样本，16条句子，两两对比8*15=120对，故总共有240条句子/batch
    if triple is True:
        label, da_embedding = [], []
        sim_list = []
        for i in range(y_pred.shape[0]):
            for j in range(i + 1, i + permute_num + 1):
                if j == y_pred.shape[0]:
                    break
                if i % 3 == 0 and j == i + 1:
                    label.append(2)
                elif i % 3 == 0 and j == i + 2 or i % 2 == 0 and j == i + 1:
                    if str(label).count('1') > noise_num:
                        continue
                    a = F.cosine_similarity(y_pred[i].unsqueeze(0), y_pred[j].unsqueeze(0), dim=-1)
                    if a.item() > 0.5:
                        continue
                    sim_list.append(a.item())
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
    else:
        label, da_embedding = [], []
        for i in range(y_pred.shape[0]):
            for j in range(i + 1, i + permute_num + 1):
                if j == y_pred.shape[0]:
                    break
                if i % 2 == 0 and j == i + 1:
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


def evaluate(model, dataloader, args):
    model.eval()
    sim_tensor = torch.tensor([], device=args.device)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(args.device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(args.device)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(args.device)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids, args.mlp_cls)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(args.device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(args.device)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(args.device)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids, args.mlp_cls)
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


def pairwise_loss(y_pred, y_true):
    sum_one = 0
    for i in range(y_pred.size(0)):
        true_index = (y_true[i] == 1.0).nonzero().flatten()
        false_index = (y_true[i] == 0.0).nonzero().flatten()
        # 若正标记有n个，负标记为m个。生成n*m矩阵，用来进行正标记与负标记的运算。
        ot = 1 - y_pred[i, true_index].view(-1, 1).repeat((1, false_index.size(0))) +\
             y_pred[i, false_index].view(1, -1).repeat((true_index.size(0), 1))
        sum_one += torch.clamp(ot, 0.0).sum()  # 将小于0的元素变为0。
    return sum_one


class TAVAT(torch.nn.Module):
    def __init__(self, model, args):
        super(TAVAT, self).__init__()
        self.model = model

    def sim_projection(self, out, adv_output):
        diff = torch.abs(out - adv_output)
        concat_vector = torch.cat([out, adv_output, diff], dim=-1)
        classify = torch.nn.Linear(768 * 3, 1).to(out.device)
        adv_sim = classify(concat_vector).squeeze(1)
        return adv_sim

    def forward(self, out, input_ids, token_type_ids, attention_mask):
        adv_inputs = {'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        # perturbation init
        embed = self.model.module.bert.embeddings.word_embeddings(input_ids)
        # delta_tok = embed.data.new(embed.size()).uniform_(-1)
        # input_ids_flat = input_ids.contiguous().view(-1)
        # gathered = torch.index_select(delta_dict, 0, input_ids_flat)  # B*seq-len D
        # delta_tok = gathered.view(embed.shape[0], embed.shape[1], -1).detach()  # B seq-len D

        # delta_tok = 6.0 * delta_tok / torch.norm(delta_tok)  # B seq-len D  normalize delta obtained from global embedding

        embed.requires_grad_()
        adv_inputs['inputs_embeds'] = embed
        adv_output = self.model(**adv_inputs)
        adv_sim = self.sim_projection(out, adv_output)
        # virtual_label = torch.ones_like(adv_sim)
        # virtual_label[::2] = 0
        label = torch.ones_like(adv_sim)
        label[::2] = label[::2] * 0.5
        adv_loss = F.mse_loss(adv_sim, label)
        # adv_loss = F.cross_entropy(adv_sim.unsqueeze(0), virtual_label.unsqueeze(0))
        delta_grad, = torch.autograd.grad(adv_loss, embed, only_inputs=True)

        # vanishing gradient
        # norm = delta_grad.norm()
        # if torch.isnan(norm) or torch.isinf(norm):
        #     return None

        # fine grain norm
        denorm_tok_ = torch.norm(delta_grad, dim=-1)  # B seq-len
        # denorm_tok = denorm_tok_.unsqueeze(2)  # B seq-len 1
        # denorm_tok = 0.2 * delta_grad / denorm_tok
        delta_tok = embed - delta_grad
        # calculate the scaling index for token gradient
        delta_norm_tok = torch.norm(delta_tok, p=2, dim=-1)  # B seq-len
        mean_norm_tok, _ = torch.max(delta_norm_tok, dim=-1, keepdim=True)  # B,1
        reweights_tok = delta_norm_tok / mean_norm_tok  # B seq-len, 1
        # top_id = list(range(1, 6))
        # random.shuffle(top_id)
        token_num = attention_mask.sum(-1)
        # top_id[0] = 10
        top_index = (token_num * 0.5).long()
        top_index = torch.eye(128)[top_index][:, :32].bool()
        b = torch.topk(denorm_tok_, 32, dim=1)[0][top_index]
        # top_token = torch.index_select(a, 1, top_index)
        filter_matrix = (denorm_tok_ >= b.unsqueeze(1)).long()
        # delta_tok = delta_tok * (reweights_tok * filter_matrix).unsqueeze(2)
        # embed[::2] = delta_tok[::2]
        # inputs_embeds = embed.detach()
        adv_inputs['inputs_embeds'], adv_inputs['output_last_hidden'] = embed, True
        # adv_inputs['inputs_embeds'] = delta_tok.detach()
        # adv_output = self.model(**adv_inputs)
        prediction_scores = self.model(**adv_inputs)
        # a = prediction_scores.view(-1, 2)
        adv_loss = F.cross_entropy(prediction_scores.view(-1, 768), filter_matrix.view(-1))
        # adv_sim = self.sim_projection(out, adv_output)
        # adv_loss = F.mse_loss(adv_sim, torch.ones_like(adv_sim) * 0.6)
        # adv_sim = F.cosine_similarity(adv_output, out)
        # adv_loss = F.pairwise_distance(adv_sim, torch.ones_like(adv_sim) * 0.4)
        # adv_loss = simcse_unsup_loss(adv_output, triple=False, loss1=False)
        # label = torch.ones_like(adv_sim)
        # label[::2] = 0
        # adv_loss = F.cross_entropy(adv_sim.unsqueeze(1), label.unsqueeze(1))
        # label[::2] = 0
        # adv_loss = F.cross_entropy(adv_sim.unsqueeze(1), label.unsqueeze(1))

################################################################################
        # out = einops.rearrange(out, '(h i) j -> h i j', i=2)
        # # adv_out = torch.cat([adv_out[::2][:adv_out.shape[0]//4], adv_out[::2][:adv_out.shape[0]//4]], dim=0)
        # out = torch.cat([out, adv_output[::2].unsqueeze(1)], dim=1)
        # all_output = einops.rearrange(out, 'h i j -> (h i) j', i=3)
        loss = simcse_unsup_loss(out, triple=False, loss1=True)
        all_loss = loss + adv_loss * 0.01
################################################################################

        # delta_tok = delta_tok.detach()
        # delta_dict = delta_dict.index_put_((input_ids_flat,), delta_tok.view(-1, 768), False)
        return all_loss  # , delta_dict


class VAT(torch.nn.Module):
    def __init__(self, model):
        super(VAT, self).__init__()
        self.model = model

    def kl(self, inputs, targets, reduction="sum"):
        """
        计算kl散度
        inputs：tensor，logits
        targets：tensor，logits
        """
        loss = F.kl_div(F.log_softmax(inputs, dim=-1),
                        F.softmax(targets, dim=-1),
                        reduction=reduction)
        return loss

    def adv_project(self, grad, norm_type='inf', eps=1e-6):
        """
        L0,L1,L2正则，对于扰动计算
        """
        if norm_type == 'l2':
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
        elif norm_type == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
        return direction

    def forward(self, hidden_status, token_type_ids, attention_mask, out):
        """
        虚拟对抗式训练
        model： nn.Module, 模型
        hidden_status：tensor，input的embedded表示
        token_type_ids：tensor，bert中的token_type_ids，A B 句子
        attention_mask：tensor，bert中的attention_mask，对paddding mask
        logits：tensor，input的输出
        """
        sim = F.cosine_similarity(out, out, dim=-1)
        sim[::2] -= 0.4
        # sim[2::4] -= 0.1
        # sim = torch.tensor(0.5).to(hidden_status.device)

        embed = hidden_status
        # 初始扰动 r
        noise = embed.data.new(embed.size()).normal_(0, 1) * 1e-5
        noise.requires_grad_()
        # x + r
        new_embed = embed.data.detach() + noise
        _, adv_output = self.model(inputs_embeds=new_embed,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask)
        # adv_logits = adv_output[0]
        # adv_sim = F.cosine_similarity(out, adv_output, dim=-1)
        diff = torch.abs(out - adv_output)
        concat_vector = torch.cat([out, adv_output, diff], dim=-1)
        classify = torch.nn.Linear(768 * 3, 1).to(out.device)
        adv_sim = classify(concat_vector)
        # adv_loss = self.kl(adv_sim, sim.detach(), reduction="batchmean")
        adv_loss = F.mse_loss(adv_sim.squeeze(1), sim)
        # noise_ = noise.repeat(1, 2, 1).view_as(hidden_status)
        delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True)
        norm = delta_grad.norm()

        # 梯度消失，退出
        if torch.isnan(norm) or torch.isinf(norm):
            return None

        # line 6 inner sum
        noise = delta_grad * 1e-3
        # line 6 projection
        noise = self.adv_project(noise, norm_type='l2', eps=1e-6)
        new_embed = embed.data.detach() + noise
        new_embed = new_embed.detach()
        # 在进行一次训练
        _, adv_output = self.model(inputs_embeds=new_embed,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask)
        # adv_logits = adv_output[0]
        return adv_output
        # adv_sim = F.cosine_similarity(out, adv_output, dim=-1)
        # return F.pairwise_distance(adv_sim, sim)
        # adv_loss_f = self.kl(adv_sim, sim.detach())
        # adv_loss_b = self.kl(sim, adv_sim.detach())
        # # 在预训练时设置为10，下游任务设置为1
        # adv_loss = (adv_loss_f + adv_loss_b) * 1
        # return adv_loss


class FGM():
    def __init__(self, model, param_name='word_embeddings', alpha=1):
        self.model = model
        self.param_name = param_name
        self.alpha = alpha
        self.data = {}

    def adversarial(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    perturbation = self.alpha * param.grad / norm
                    param.data.add_(perturbation)

    def backup_param_data(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                self.data[name] = param.data.clone()

    def restore_param_data(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                assert name in self.data
                param.data = self.data[name]

    def adversarial_training(self, inputs, args):
        self.backup_param_data()
        self.adversarial()
        _, out = self.model(**inputs)
        loss = simcse_unsup_loss(out, triple=args.multi_dropout, loss1=False)
        loss.backward()
        self.restore_param_data()


def train(model, tokenizer, writer, args):
    train_dataset = PrepareDataset(args, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataset = PrepareDataset(args, tokenizer, 'dev')
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)
    optimizer, scheduler = scheduler_with_optimizer(model, train_loader, args)
    # adv = FGM(model)
    best = 0
    for epoch in range(args.num_epochs):
        model.train()
        model.zero_grad()
        TAVAT_model = TAVAT(model, args)
        # delta_dict = torch.zeros([30522, 768]).uniform_(-1).to(args.device)
        # dims = torch.tensor([768]).float()  # (768^(1/2))
        # mag = 0.1 / torch.sqrt(dims)  # 1 const (small const to init delta)
        # delta_dict = (delta_global_embedding * mag.view(1, 1))
        early_stop = 0
        for batch_idx, data in enumerate(tqdm(train_loader)):
            label = None
            if len(data) == 2:
                data, score = data[0], data[1]
                label = score.float().to(args.device)
            step = epoch * len(train_loader) + batch_idx
            sql_len = data['input_ids'].shape[-1]
            inputs = {'input_ids': data['input_ids'].view(-1, sql_len).to(args.device),
                      'attention_mask': data['attention_mask'].view(-1, sql_len).to(args.device),
                      'token_type_ids': data['token_type_ids'].view(-1, sql_len).to(args.device)}
            out = model(**inputs)
            # out_ = model(input_ids[::2], attention_mask[::2], token_type_ids[::2], plus=True)
            # out = einops.rearrange(out, '(h i) j -> h i j', i=2)
            # out = torch.cat([out, out_.unsqueeze(1)], dim=1)
            # out = einops.rearrange(out, 'h i j -> (h i) j', i=3)
            # del out_
            if args.loss == 'CL':
                if args.adv == True:
                    inputs['out'] = out
                    loss = TAVAT_model(**inputs)
                    # delta_global_embedding = delta_embed_dict

                    # out = einops.rearrange(out, '(h i) j -> h i j', i=2)
                    # # adv_out = torch.cat([adv_out[::2][:adv_out.shape[0]//4], adv_out[::2][:adv_out.shape[0]//4]], dim=0)
                    # out = torch.cat([out, adv_out[::2].unsqueeze(1)], dim=1)
                    # out = einops.rearrange(out, 'h i j -> (h i) j', i=3)

                    # out = out.repeat(1, 2).view(64 * 4, -1)
                    # out[2::4], out[3::4] = adv_out[::2], adv_out[1::2]
                    # loss = simcse_unsup_loss(out, triple=True, loss1=True)

                    # alpha = 0.0001
                    # loss = loss + adv_out * alpha
                    # out = einops.rearrange(out, '(h i) j -> h i j', i=2)
                    # loss = prompt_loss(out[:, 0, :], out[:, 1, :])
                else:
                    loss = simcse_unsup_loss(out, triple=False, loss1=True)
            elif args.loss == 'CS':
                out_list, labels = da_function(out, args.device, args.permute_num, args.noise_num, triple=args.multi_dropout)
                loss = cosent_loss(out_list, labels)
            elif args.loss == 'SCD':
                out = einops.rearrange(out, '(h i) j -> h i j', h=2)
                loss = SCD_Loss(args)(out[0, :, :], out[1, :, :])
            elif args.loss == 'SUP_STS':
                # out = einops.rearrange(out, '(h i) j -> h i j', i=2)
                # sim = F.cosine_similarity(out[:, 0, :], out[:, 1, :], dim=-1)
                # sim = F.sigmoid(sim)
                # loss = F.mse_loss(sim, label)
                loss = cosent_loss(out, label)
            elif args.loss == 'SUP_NLI':
                loss = simcse_sup_loss(out)
            elif args.loss == 'SNCSE':
                # out = einops.rearrange(out, '(h i) j -> h i j', i=3)
                loss = SNCSE_loss(out)
            elif args.loss == "PL":
                loss = prompt_loss(out[::2], out[1::2])
            else:
                loss = None
            # loss /= args.gradient_accumulation_step

            # adv.adversarial_training(inputs, args)
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
                # early_stop += 1
                # if early_stop == 40:
                #     logger.info(f"corrcoef doesn't improve for {early_stop} batch, early stop!")
                #     logger.info(f"train use sample number: {(batch_idx - 10) * args.batch_size}")
                #     logger.info('dev_best_corrcoef: {}'.format(best))
                #     return


def test(model, tokenizer, args):
    test_dataset = PrepareDataset(args, tokenizer, 'test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    model.load_state_dict(torch.load(join(args.output_path, 'bert.pt')))
    corrcoef = evaluate(model, test_loader, args)
    logger.info('test corrcoef:{}'.format(corrcoef))


def SentEval(model, tokenizer, args):
    model.load_state_dict(torch.load(join(args.output_path, 'bert.pt')))
    result_STS, result_transfer = do_senteval(tokenizer, model, args.device, args.mlp_cls)
    logger.info('\n{}'.format(result_STS))


if __name__ == '__main__':
    for seed in range(57, 150):
        main(seed)
