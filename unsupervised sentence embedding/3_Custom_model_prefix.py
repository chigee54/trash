import random

import numpy
import torch
import torch.nn as nn
from transformers import AutoConfig, BertPreTrainedModel
from modeling_bert import BertModel, BertLMPredictionHead, BertForMaskedLM
import torch.nn.functional as F
import einops
import json
import numpy as np


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)
        return x


class PrefixEncoder(nn.Module):
    r'''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''

    def __init__(self, config, pre_seq_len=16, prefix_projection=False):
        super().__init__()
        self.prefix_projection = prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(torch.nn.Linear(config.hidden_size, config.hidden_size), torch.nn.Tanh(),
                         torch.nn.Linear(config.hidden_size, config.num_hidden_layers * 2 * config.hidden_size))
        else:
            self.embedding = torch.nn.Embedding(pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class Unilm_loss(nn.Module):
    def __init__(self, cls, seq, ids):
        super().__init__()
        self.compute_loss(cls, seq, ids)

    def compute_loss(self, outputs_cls, outputs_seq, pt_batch):
        loss_sim, correct_sim = self.compute_sim_loss(outputs_cls)
        loss_seq, correct_seq, denominator_seq = self.compute_seq_loss(outputs_seq, pt_batch)
        loss = loss_seq + loss_sim
        return loss, loss_seq, loss_sim, correct_seq, correct_sim, denominator_seq

    def compute_seq_loss(self, outputs_seq, pt_batch):
        y_pred = outputs_seq[:, :-1, :]
        y_true = pt_batch['input_ids'][:, 1:]
        y_mask = pt_batch['token_type_ids'][:, 1:]  # segment embedding
        y_pred = y_pred[y_mask > 0, :]  # 错位预测,只预测第二句话（即segment_id=1）
        y_true = y_true[y_mask > 0]  # 错位预测
        y_mask = y_mask[y_mask > 0]
        denominator_seq = torch.sum(y_mask)
        loss = F.cross_entropy(y_pred, y_true)
        correct_seq = torch.sum(torch.eq(torch.max(y_pred, dim=1)[1], y_true))
        return loss, correct_seq, denominator_seq

    def compute_sim_loss(self, outputs_cls):
        y_true = self.get_sim_label(outputs_cls).to(device)
        y_pred = F.normalize(outputs_cls, p=2, dim=1)
        similarities = torch.mm(y_pred, y_pred.t())
        similarities = similarities - (torch.eye(y_pred.shape[0]) * 1e12).to(device)  # 排除对角线
        similarities = similarities * 30  # scale

        index = [i for i in range(outputs_cls.shape[0])]
        np.random.shuffle(index)
        y_true = y_true[index]
        similarities = similarities[index]
        loss = F.cross_entropy(similarities, y_true)

        y_hat = torch.max(similarities, 1)[1]
        correct_sim = torch.sum(torch.eq(y_hat, y_true))

        return loss, correct_sim

    def get_sim_label(self, outputs_cls):
        idxs = torch.arange(0, outputs_cls.shape[0])
        idxs_2 = (idxs + 1 - idxs % 2 * 2)
        return idxs_2


class Custom_Pretrained_Model(BertPreTrainedModel):
    """模型自定义"""
    def __init__(self, config, **model_kargs):
        super(Custom_Pretrained_Model, self).__init__(config)
        if model_kargs['multi_dropout']:
            config.multi_dropout = model_kargs['multi_dropout']
            config.attention_probs_dropout_prob_noise = model_kargs['dropout_noise']
            config.hidden_dropout_prob_noise = model_kargs['dropout_noise']
        config.attention_probs_dropout_prob = model_kargs['dropout']
        config.hidden_dropout_prob = model_kargs['dropout']
        # self.mlm = BertForMaskedLM(config)
        # self.mlm_orig = BertForMaskedLM.from_pretrained('pretrained_model/bert_based_uncased_english', config=config)
        # self.mlm_predict = BertLMPredictionHead(config)
        self.config = config
        self.classification = model_kargs['classification']
        self.bert_two = BertModel(config)
        self.bert = BertModel(config)
        self.mlp = MLPLayer(config)
        self.pooling = model_kargs['pooling']
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, model_kargs['num_labels'])   # the num_labels of STS-B is 1
        self.do_ptuning = model_kargs['do_ptuning']
        self.init_weights()
        ##########################################################################
        if self.do_ptuning:
            self.pre_seq_len = model_kargs['pre_seq_len']
            self.n_layer = config.num_hidden_layers
            self.n_head = config.num_attention_heads
            self.n_embd = config.hidden_size // config.num_attention_heads
            self.prefix_tokens = torch.arange(model_kargs['pre_seq_len']).long()

            self.prefix_encoder = PrefixEncoder(config, pre_seq_len=model_kargs['pre_seq_len'], prefix_projection=model_kargs['prefix_projection'])
            ################
            # self.pre_seq_len_plus = pre_seq_len_plus
            # self.prefix_tokens_plus = torch.arange(pre_seq_len_plus).long()
            # self.prefix_encoder_plus = PrefixEncoder(config, pre_seq_len=pre_seq_len_plus, prefix_projection=prefix_projection)
            ################
            self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

            for param in self.bert.parameters():
                param.requires_grad = model_kargs['train_bert']
            for param in self.prefix_encoder.parameters():
                param.requires_grad = True

            # compute the number of total parameters and tunable parameters
            total_param = sum(p.numel() for p in self.parameters())
            trainable_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print('total param is {}, trainable param is {}, Rate is {}'.format(total_param, trainable_param, trainable_param/total_param))
        ##########################################################################

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, mlp_cls=True, do_mlm=False, inputs_embeds=None, past_key_values=None):

        ##########################################################################
        if self.do_ptuning and past_key_values is None:
            # concatenate prefix_key and prefix_value with original key and value in hidden states
            past_key_values = self.get_prompt(batch_size=attention_mask.shape[0])
            prefix_attention_mask = torch.ones(attention_mask.shape[0], self.pre_seq_len).to(self.bert.device)
            # prefix_attention_mask[2::3][0] = 0
            # a = attention_mask[2::3]
            # filter_matrix = []
            # for i in range(a.shape[0]):
            #     num_tokens = a[i].sum().int().item()
            #     index = list(range(1, num_tokens-1))
            #     random.shuffle(index)
            #     if num_tokens < 10:
            #         c = a[i].index_fill(0, torch.tensor(index[0]).to(self.bert.device), 0.0) if num_tokens != 2 else a[i]
            #     elif num_tokens < 20:
            #         c = a[i].index_fill(0, torch.tensor(index[:2]).to(self.bert.device), 0.0)
            #     else:
            #         c = a[i].index_fill(0, torch.tensor(index[:3]).to(self.bert.device), 0.0)
            #     filter_matrix.append(c.unsqueeze(0))
            # attention_mask[2::3] = torch.cat(filter_matrix, 0)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        ##########################################################################
        # [last_hidden_state, pooler_output, hidden_states, attentions]
        out = self.bert(input_ids, attention_mask, token_type_ids, past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds, output_hidden_states=True)
        if do_mlm:
            #####################################################
            # pooler_output = out.pooler_output
            # input_ids = torch.zeros_like(input_ids).cuda()
            # new_output = self.bert_two(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            #                            cls_input=pooler_output.view((-1, pooler_output.size(-1))))
            # projector = torch.nn.Linear(768, 30522).cuda()
            # new_output = projector(new_output[0])
            #####################################################
            pooler_output = out.pooler_output
            # add_id = torch.ones_like(input_ids[:, 0]).unsqueeze(1)
            # input_ids = torch.cat([input_ids[:, 0].unsqueeze(1), add_id.cuda() * 102, input_ids[:, 1:]], dim=1)
            # token_type_ids = torch.cat([add_id * 0, add_id * 0, torch.ones_like(token_type_ids[:, 1:])], dim=1)
            input_ids = torch.cat([input_ids, input_ids[:, 1:]], dim=1)
            token_type_ids = torch.cat([token_type_ids, token_type_ids[:, 1:] + 1], dim=1)
            new_ids = [input_ids, token_type_ids]
            attention_mask = self.compute_attention_bias(token_type_ids)
            # padding_mask = (input_ids != 0).to(input_ids.device).unsqueeze(0).expand(self.config.num_attention_heads, token_type_ids.shape[0], token_type_ids.shape[1])
            new_output = self.bert_two(input_ids=input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,)
                                       # head_mask=padding_mask.unsqueeze(0).expand(self.config.num_attention_heads, token_type_ids.size(0), token_type_ids.size(1)),
                                       # cls_input=pooler_output.view((-1, pooler_output.size(-1))))
            projector = torch.nn.Linear(768, 30522).cuda()
            new_output = projector(new_output[0])
            return pooler_output, new_output, new_ids
            # return self.mlm_predict(out[0])
        if self.classification:
            pooled_output = self.dropout(out[1])
            logits = self.classifier(pooled_output)
            return (logits,) + out[2:]  # add hidden states and attention if they are here
        if self.pooling == 'cls':
            # out[0] is last_hidden_state
            # a = out[2][0]
            # b = out[3][0][:, :, :, :16]
            # a = self.mlp(out.last_hidden_state[:, 0])
            return self.mlp(out.last_hidden_state[:, 0]) if mlp_cls is True else out.last_hidden_state[:, 0]
            # if mlp_cls is True:
            #     final_out = self.mlp(out.last_hidden_state[:, 0]).cpu()
            #     tokens_out = out.last_hidden_state[:, :].transpose(1, 2)
            #     avg_tokens = torch.avg_pool1d(tokens_out, kernel_size=tokens_out.shape[-1]).squeeze(-1)
            #     avg_tokens = self.mlp(avg_tokens).cpu()
            #     final_out[2::3] = avg_tokens[2::3].cpu()
            #     return final_out.to(self.bert.device)
            # else:
            #     return out.last_hidden_state[:, 0]
        if self.pooling == 'pooler':
            # out[1] is pooler_output
            return out.pooler_output  # [batch, 768]
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]

    def compute_attention_bias(self, segment_ids):
        idxs = torch.cumsum(segment_ids, dim=1)
        mask = idxs[:, None, :] <= idxs[:, :, None]
        mask = mask.float()
        mask = -(1.0 - mask) * 1e12
        return mask

    ##########################################################################
    def get_prompt(self, batch_size):
        # if plus is False:
            # change into the format needed for bert
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(batch_size, -1, self.n_layer * 2, self.n_head, self.n_embd)
        past_key_values = self.dropout(past_key_values)
        # transpose dimension
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
        # else:
        #     # change into the format needed for bert
        #     prefix_tokens = self.prefix_tokens_plus.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        #     past_key_values = self.prefix_encoder_plus(prefix_tokens)
        #     # bsz, seqlen, _ = past_key_values.shape
        #     past_key_values = past_key_values.view(batch_size, self.pre_seq_len_plus, self.n_layer * 2, self.n_head,
        #                                            self.n_embd)
        #     past_key_values = self.dropout(past_key_values)
        #     # transpose dimension
        #     past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        #     return past_key_values, self.pre_seq_len_plus
    ##########################################################################


class _Custom_Model(nn.Module):
    """模型自定义"""
    def __init__(self, pretrained_model, num_labels=1, dropout=0.1, multi_dropout=False, dropout_noise=None):
        super(_Custom_Model, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model)
        if multi_dropout:
            config.multi_dropout = multi_dropout
            config.attention_probs_dropout_prob_noise = dropout_noise
            config.hidden_dropout_prob_noise = dropout_noise
        config.attention_probs_dropout_prob = dropout
        config.hidden_dropout_prob = dropout
        # self.mlm_predict = BertLMPredictionHead(config)
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.mlp = MLPLayer(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)   # the num_labels of STS-B is 1

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, mlp_cls=True):

        # [last_hidden_state, pooler_output, hidden_states, attentions]
        out = self.bert(input_ids, attention_mask, token_type_ids)
        return self.mlp(out.last_hidden_state[:, 0]) if mlp_cls is True else out.last_hidden_state[:, 0]


class CircleLoss(nn.Module):
    def __init__(self, scale=32, margin=0.25, similarity='cos', **kwargs):
        super(CircleLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.similarity = similarity

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"

        m = labels.size(0)
        mask = labels.expand(m, m).t().eq(labels.expand(m, m)).float()
        pos_mask = mask.triu(diagonal=1)
        neg_mask = (mask - 1).abs_().triu(diagonal=1)
        if self.similarity == 'dot':
            sim_mat = torch.matmul(feats, torch.t(feats))
        elif self.similarity == 'cos':
            feats = F.normalize(feats)
            sim_mat = feats.mm(feats.t())
        else:
            raise ValueError('This similarity is not implemented.')

        pos_pair_ = sim_mat[pos_mask == 1]
        neg_pair_ = sim_mat[neg_mask == 1]

        alpha_p = torch.relu(-pos_pair_ + 1 + self.margin)
        alpha_n = torch.relu(neg_pair_ + self.margin)
        margin_p = 1 - self.margin
        margin_n = self.margin
        loss_p = torch.sum(torch.exp(-self.scale * alpha_p * (pos_pair_ - margin_p)))
        loss_n = torch.sum(torch.exp(self.scale * alpha_n * (neg_pair_ - margin_n)))
        loss = torch.log(1 + loss_p * loss_n)
        return loss


class Custom_Model(nn.Module):
    """模型自定义"""
    def __init__(self, pretrained_model, pooling='cls', pre_seq_len=16, num_labels=1, dropout=0.1, multi_dropout=False,
                 dropout_noise=None, do_ptuning=False, prefix_projection=False, classification=False, train_bert=False):
        super(Custom_Model, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model)
        if multi_dropout is not None:
            config.multi_dropout = multi_dropout
            config.attention_probs_dropout_prob_noise = dropout_noise
            config.hidden_dropout_prob_noise = dropout_noise
        config.attention_probs_dropout_prob = dropout
        config.hidden_dropout_prob = dropout
        # self.mlm = BertForMaskedLM.from_pretrained(pretrained_model, config=config)
        self.bert_two = BertModel(config)
        self.mlm_predict = BertLMPredictionHead(config)
        self.classification = classification
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.mlp = MLPLayer(config)
        self.pooling = pooling
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)   # the num_labels of STS-B is 1
        self.do_ptuning = do_ptuning
        self.antonym_dict = json.loads(open('datasets/synonym_antonym_clean.json', 'r', encoding='utf8').read())

        ##########################################################################
        if self.do_ptuning:
            self.pre_seq_len = pre_seq_len
            self.n_layer = config.num_hidden_layers
            self.n_head = config.num_attention_heads
            self.n_embd = config.hidden_size // config.num_attention_heads
            self.prefix_tokens = torch.arange(pre_seq_len).long()

            self.prefix_encoder = PrefixEncoder(config, pre_seq_len=pre_seq_len, prefix_projection=prefix_projection)
            ################
            # self.pre_seq_len_plus = pre_seq_len_plus
            # self.prefix_tokens_plus = torch.arange(pre_seq_len_plus).long()
            # self.prefix_encoder_plus = PrefixEncoder(config, pre_seq_len=pre_seq_len_plus, prefix_projection=prefix_projection)
            ################
            self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

            for param in self.bert.parameters():
                param.requires_grad = train_bert
            for param in self.prefix_encoder.parameters():
                param.requires_grad = True

            # compute the number of total parameters and tunable parameters
            total_param = sum(p.numel() for p in self.parameters())
            trainable_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print('total param is {}, trainable param is {}, Rate is {}'.format(total_param, trainable_param, trainable_param/total_param))
        ##########################################################################

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, mlp_cls=True, do_mlm=False,
                dual_output=False, inputs_embeds=None, past_key_values=None):

        ##########################################################################
        if self.do_ptuning and past_key_values is None:
            # concatenate prefix_key and prefix_value with original key and value in hidden states
            past_key_values = self.get_prompt(batch_size=attention_mask.shape[0])
            prefix_attention_mask = torch.ones(attention_mask.shape[0], self.pre_seq_len).to(self.bert.device)
            # prefix_attention_mask[2::3][0] = 0
            # a = attention_mask[2::3]
            # filter_matrix = []
            # for i in range(a.shape[0]):
            #     num_tokens = a[i].sum().int().item()
            #     index = list(range(1, num_tokens-1))
            #     random.shuffle(index)
            #     if num_tokens < 10:
            #         c = a[i].index_fill(0, torch.tensor(index[0]).to(self.bert.device), 0.0) if num_tokens != 2 else a[i]
            #     elif num_tokens < 20:
            #         c = a[i].index_fill(0, torch.tensor(index[:2]).to(self.bert.device), 0.0)
            #     else:
            #         c = a[i].index_fill(0, torch.tensor(index[:3]).to(self.bert.device), 0.0)
            #     filter_matrix.append(c.unsqueeze(0))
            # attention_mask[2::3] = torch.cat(filter_matrix, 0)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        ##########################################################################
        # [last_hidden_state, pooler_output, hidden_states, attentions]
        out = self.bert(input_ids, attention_mask, token_type_ids, past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds, output_hidden_states=True)
        if do_mlm:
            # input_ids = self._get_masked(input_ids)
            # pooler_output = out.pooler_output
            # prediction_scores = self.mlm(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            #                     cls_input=pooler_output.view((-1, pooler_output.size(-1))))[0]
            # return self.mlm_process(pooler_output, prediction_scores)
            a = torch.cat((out[1][:, 0].unsqueeze(1), torch.zeros_like(out[0][:, 1:])), dim=1)

            return out[1], self.mlm_predict(a)
        # if bert_ouput:
        #     return out
        if self.classification:
            pooled_output = self.dropout(out[1])
            logits = self.classifier(pooled_output)
            return (logits,) + out[2:]  # add hidden states and attention if they are here
        if dual_output:
            return self.mlp(out[0][:, 0]) if mlp_cls is True else out[0][:, 0], out.hidden_states[-2]
        if self.pooling == 'cls':
            # out[0] is last_hidden_state
            # a = out[2][0]
            # b = out[3][0][:, :, :, :16]
            # a = self.mlp(out.last_hidden_state[:, 0])
            return self.mlp(out[0][:, 0]) if mlp_cls is True else out[0][:, 0]
            # if mlp_cls is True:
            #     final_out = self.mlp(out.last_hidden_state[:, 0]).cpu()
            #     tokens_out = out.last_hidden_state[:, :].transpose(1, 2)
            #     avg_tokens = torch.avg_pool1d(tokens_out, kernel_size=tokens_out.shape[-1]).squeeze(-1)
            #     avg_tokens = self.mlp(avg_tokens).cpu()
            #     final_out[2::3] = avg_tokens[2::3].cpu()
            #     return final_out.to(self.bert.device)
            # else:
            #     return out.last_hidden_state[:, 0]
        if self.pooling == 'pooler':
            # out[1] is pooler_output
            return out.pooler_output  # [batch, 768]
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]

    ##########################################################################
    def get_prompt(self, batch_size):
        # if plus is False:
            # change into the format needed for bert
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(batch_size, -1, self.n_layer * 2, self.n_head, self.n_embd)
        past_key_values = self.dropout(past_key_values)
        # transpose dimension
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
        # else:
        #     # change into the format needed for bert
        #     prefix_tokens = self.prefix_tokens_plus.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        #     past_key_values = self.prefix_encoder_plus(prefix_tokens)
        #     # bsz, seqlen, _ = past_key_values.shape
        #     past_key_values = past_key_values.view(batch_size, self.pre_seq_len_plus, self.n_layer * 2, self.n_head,
        #                                            self.n_embd)
        #     past_key_values = self.dropout(past_key_values)
        #     # transpose dimension
        #     past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        #     return past_key_values, self.pre_seq_len_plus

    def _get_masked(self, input_ids):
        # a = self.antonym_dict.keys()
        antonym_list = [int(i) for i in self.antonym_dict.keys()]
        filter_matrix = np.matrix([(input_ids == antonym) for antonym in antonym_list]).sum()
        mask_input_ids = torch.where(filter_matrix == 1, 103, input_ids)
        return mask_input_ids

    def get_candidates(self, input_ids):
        # a = self.antonym_dict.keys()
        antonym_list = [int(i) for i in self.antonym_dict.keys()]
        filter_matrix = np.matrix([(input_ids == antonym) for antonym in antonym_list]).sum()
        select_antonym = input_ids[filter_matrix]
        len_candi = []
        candi_list = []
        for antonym in select_antonym:
            candi = self.antonym_dict[str(antonym.item())]
            candi_list.extend(candi)
            len_candi.append(len(candi))
        return filter_matrix, torch.tensor(candi_list).cuda(), torch.tensor(len_candi).cuda()

    def mlm_process(self, out, prediction_scores):
        word_predictions = torch.max(prediction_scores, dim=-1)[1]
        filter_matrix, candidates, len_candi = self.get_candidates(word_predictions)
        filter_embed = torch.repeat_interleave(word_predictions[filter_matrix].unsqueeze(1), max(len_candi), dim=1)

        expand_matrix = (torch.arange(0, max(len_candi)).unsqueeze(0).cuda() < len_candi.unsqueeze(1))
        filter_embed = filter_embed[expand_matrix]
        filter_embed = self.bert.embeddings.word_embeddings(filter_embed)
        candidates_embed = self.bert.embeddings.word_embeddings(candidates)
        all_scores = F.cosine_similarity(filter_embed, candidates_embed)
        adv_loss = F.mse_loss(all_scores, torch.ones_like(all_scores) * 0.4)
        loss = simcse_unsup_loss(out, triple=False, loss1=True)
        loss = loss + adv_loss * 0.01
        return loss
    ##########################################################################


from scipy.sparse import block_diag


def simcse_unsup_loss(y_pred, triple=True, temp=0.05, loss1=True, extra_label=False):
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]
    """
    device = y_pred.device
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    if triple:
        # a = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        a = [[0, 1, 0.5], [0.5, 0, 1], [0.5, 0.5, 0]]
        # a = [[0, 3, 2, 1], [3, 0, 2, 1], [2, 2, 0, 1], [1, 1, 1, 0]]
        y_true = block_diag([a for _ in range(y_pred.shape[0]//3)]).toarray()
        y_true = torch.Tensor(y_true).to(device)

        # c = (y_true != 2).type(torch.long)
    else:
        if extra_label is False:
            # y_true = torch.arange(y_pred.shape[0], device=device)
            # y_true = (y_true - y_true % 2 * 2) + 1
            a = [[0, 1], [1, 0]]
            y_true = block_diag([a for _ in range(y_pred.shape[0] // 2)]).toarray()
            y_true = torch.Tensor(y_true).to(device)
        else:
            a = [[0, 0.5], [0.5, 0]]
            y_true = block_diag([a for _ in range(y_pred.shape[0] // 2)]).toarray()
            y_true = torch.Tensor(y_true).to(device)
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # a, b = torch.max(sim), torch.min(sim)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    # sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    # filter_matrix = torch.tril(torch.ones_like(sim), 0)
    # sim = sim - filter_matrix.to(device) * 1e12
    # if triple:
    #     y = (sim > 0.8).type(torch.long)
    #     z = torch.where(c == 0, c, y)
    #     sim = sim - z.to(device) * 1e12
    # 相似度矩阵除以温度系数
    # sim = sim / temp

    # right = torch.cat([torch.diag(sim[:, 1:]), torch.diag(sim[:, 2:])], dim=0)
    # filter_sim = torch.cat([right, torch.diag(sim[:, 3:])], dim=0)
    # filter_sim = torch.cat([filter_sim, torch.diag(sim[:, 4:])], dim=0)
    # filter_sim = torch.cat([filter_sim, torch.diag(sim[:, 5:])], dim=0)
    # filter_sim = torch.cat([filter_sim, torch.diag(sim[:, 6:])], dim=0)
    # left = torch.cat([torch.diag(sim[1:]), torch.diag(sim[2:])], dim=0)
    # filter_sim = torch.cat([right, left], dim=0)
    # right_ = torch.cat([torch.diag(y_true[:, 1:]), torch.diag(y_true[:, 2:])], dim=0)
    # filter_label = torch.cat([right_, torch.diag(y_true[:, 3:])], dim=0)
    # filter_label = torch.cat([filter_label, torch.diag(y_true[:, 4:])], dim=0)
    # filter_label = torch.cat([filter_label, torch.diag(y_true[:, 5:])], dim=0)
    # filter_label = torch.cat([filter_label, torch.diag(y_true[:, 6:])], dim=0)
    # left_ = torch.cat([torch.diag(y_true[1:]), torch.diag(y_true[2:])], dim=0)
    # filter_label = torch.cat([right_, left_], dim=0)
    # filter_sim, filter_label = torch.triu(sim, 1).view(-1), torch.triu(y_true, 1).view(-1)

    # 计算相似度矩阵与y_true的交叉熵损失
    # 计算交叉熵，每个case都会计算与其他case的相似度得分，得到一个得分向量，目的是使得该得分向量中正样本的得分最高，负样本的得分最低
    if loss1:
        sim = sim / temp
        filter_matrix = torch.tril(torch.triu(torch.ones_like(sim), 1), 25).bool()
        filter_sim, filter_label = sim[filter_matrix], y_true[filter_matrix]
        # filter_sim, filter_label = sim - (1 - filter_matrix) * 1e12, y_true - (1 - filter_matrix) * 1e12
        loss = cosent(filter_sim, filter_label)
        # loss = pairwise_loss(filter_sim, filter_label)
        # loss = CircleLoss(similarity='sub')(filter_sim, filter_label)
    else:
        sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
        # sim = sim - torch.tril((y_true == 1).type(torch.float).flip(1), 0).flip(1) * 1e12
        sim = sim / temp
        loss = torch.mean(F.cross_entropy(sim, y_true))

    # loss = torch.mean(F.binary_cross_entropy_with_logits(sim, y_true))
    return loss


def pairwise_loss(y_pred, y_true):
    true_index = (y_true == 1.0).nonzero().flatten()
    false_index = (y_true == 0.0).nonzero().flatten()
    # 若正标记有n个，负标记为m个。生成n*m矩阵，用来进行正标记与负标记的运算。
    a = y_pred[true_index].view(-1, 1)
    a = a.repeat((1, false_index.size(0)))
    b = y_pred[false_index].view(1, -1)
    b = b.repeat((true_index.size(0), 1))
    ot = 0.5 - a + b
    sum_one = torch.clamp(ot, 0.0).sum()  # 将小于0的元素变为0。
    return sum_one


# class CircleLoss(nn.Module):
#     def __init__(self, scale=20, margin=0.5, similarity='cos', **kwargs):
#         super(CircleLoss, self).__init__()
#         self.scale = scale
#         self.margin = margin
#         self.similarity = similarity
#
#     def forward(self, feats, labels):
#         assert feats.size(0) == labels.size(0), f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
#         y_true = (labels[:, None] < labels[None, :]).float()
#         # m = labels.size(0)
#         # mask = labels.expand(m, m).t().eq(labels.expand(m, m)).float()
#         # pos_mask = mask.triu(diagonal=1)
#         # neg_mask = (mask - 1).abs_().triu(diagonal=1)
#         if self.similarity == 'dot':
#             sim_mat = torch.matmul(feats, torch.t(feats))
#         elif self.similarity == 'cos':
#             feats = F.normalize(feats)
#             sim_mat = feats.mm(feats.t())
#         elif self.similarity == 'sub':
#             sim_mat = feats[:, None] - feats[None, :]
#         else:
#             raise ValueError('This similarity is not implemented.')
#
#         # pos_pair_ = sim_mat[y_true == 1.0]
#         pos_pair_ = sim_mat - (1 - y_true) * 1e12
#         y_pred = pos_pair_.view(-1)
#         y_pred = torch.cat((torch.tensor([0]).float().cuda(), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
#         return torch.logsumexp(y_pred, dim=0)
#         # alpha_p = torch.relu(-pos_pair_ + self.margin)
#         # loss_p = torch.sum(torch.exp(-alpha_p * pos_pair_))
#         # loss = torch.log(1 + loss_p)
#         # return loss


def cosent(y_pred, y_true, margin=0.25):
    # ap = torch.clamp_min(- y_pred.unsqueeze(0).detach() + 1 + margin, min=0.)
    # an = torch.clamp_min(y_pred.unsqueeze(1).detach() + margin, min=0.)
    # ap = torch.relu(-y_pred + 1 + margin)
    # an = torch.relu(y_pred + margin)
    y_pred = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)  # 这里是算出所有位置 两两之间余弦的差值
    y_true = y_true.unsqueeze(1) < y_true.unsqueeze(0)  # 取出负例-正例的差值
    y_true = y_true.float()
    y_pred = y_pred - (1 - y_true) * 1e12
    y_pred = y_pred.view(-1)
    # y_pred = y_pred[y_true]
    y_pred = torch.cat((torch.tensor([0]).float().cuda(), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
    return torch.logsumexp(y_pred, dim=0)


def simcse_sup_loss(y_pred: 'tensor') -> 'tensor':
    """有监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 3, 768]
    """
    device = y_pred.device
    # 得到y_pred对应的label, 每第三句没有label, 跳过, label= [1, 0, 4, 3, ...]
    y_true = torch.arange(y_pred.shape[0], device=device)
    use_row = torch.where((y_true + 1) % 3 != 0)[0]
    y_true = (use_row - use_row % 3 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    # 选取有效的行
    sim = torch.index_select(sim, 0, use_row)
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss


def prompt_loss(query, key, tao=0.05):
    query = torch.div(query, torch.norm(query, dim=1).reshape(-1, 1))
    key = torch.div(key, torch.norm(key, dim=1).reshape(-1, 1))
    # print(query.shape, key.shape)
    N, D = query.shape[0], query.shape[1]
    # calculate positive similarity
    batch_pos = torch.exp(torch.div(torch.bmm(query.view(N, 1, D), key.view(N, D, 1)).view(N, 1), tao))
    # calculate inner_batch all similarity
    batch_all = torch.sum(torch.exp(torch.div(torch.mm(query.view(N, D), torch.t(key)), tao)), dim=1)
    loss = torch.mean(-torch.log(torch.div(batch_pos, batch_all)))
    return loss


def InfoNCE_loss(rep1, rep2, temperature=0.05):
    normalized_rep1 = F.normalize(rep1)
    normalized_rep2 = F.normalize(rep2)
    dis_matrix = torch.mm(normalized_rep1, normalized_rep2.T) / temperature
    pos = torch.diag(dis_matrix)
    dedominator = torch.sum(torch.exp(dis_matrix), dim=1)
    loss = (torch.log(dedominator) - pos).mean()
    return loss


def SNCSE_loss(out, alpha=0.3, beta=0.7, lambda_=0.001):
    # Calculate InfoNCE loss
    # loss = prompt_loss(z1, z2)
    out = einops.rearrange(out, '(h i) j -> h i j', i=3)
    z1, z2, z3 = out[:, 0, :], out[:, 1, :], out[:, 2, :]
    loss = simcse_unsup_loss(einops.rearrange(out[:, [0, 1], :], 'h i j -> (h i) j'))
    temp1 = torch.cosine_similarity(z1, z2, dim=1)  # Cosine similarity of positive pairs
    temp2 = torch.cosine_similarity(z1, z3, dim=1)  # Cosine similarity of soft negative pairs
    temp = temp2 - temp1  # similarity difference
    loss1 = torch.relu(temp + alpha) + torch.relu(-temp - beta)  # BML loss
    loss1 = torch.mean(loss1)
    loss += loss1 * lambda_  # Total loss
    return loss


def cosent_loss(y_pred, y_true, temp=20):
    norms = (y_pred ** 2).sum(axis=1, keepdims=True) ** 0.5
    y_pred = y_pred / norms
    y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1)
    pn_index = (y_true > 0).long()
    y_pred = y_pred * temp  # * (pn_index * 0.5)
    y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置 两两之间余弦的差值
    y_true = y_true[:, None] < y_true[None, :]  # 取出负例-正例的差值
    y_true = y_true.float()
    y_pred = y_pred - (1 - y_true) * 1e12
    y_pred = y_pred.view(-1)
    y_pred = torch.cat((torch.tensor([0]).float().cuda(), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
    return torch.logsumexp(y_pred, dim=0)


class SCD_Loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.alpha = args.alpha
        self.beta = args.beta
        self.lambd = args.lambd
        self.temp = args.temp
        self.sizes = [args.embedding_dim] + list(map(int, args.projector_num.split('-')))
        self.sim = nn.CosineSimilarity(dim=-1)

    def forward(self, z1, z2):
        bn = nn.BatchNorm1d(self.sizes[-1], affine=False).cuda()
        c = bn(self.projector()(z1)).T @ bn(self.projector()(z2))
        # sum the cross-correlation matrix between all gpus
        c.div_(len(z1))
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        # return a flattened view of the off-diagonal elements of a square matrix
        off_diagonal = c.flatten()[:-1].view(c.shape[0] - 1, c.shape[0] + 1)[:, 1:].flatten()
        off_diag = off_diagonal.pow_(2).sum()
        decorrelation = on_diag + self.lambd * off_diag
        self_contrast = torch.diag(self.sim(z1.unsqueeze(1), z2.unsqueeze(0)) / self.temp).mean()
        loss = self.alpha * self_contrast + self.beta * decorrelation
        return loss

    def projector(self):
        layers = []
        for i in range(len(self.sizes) - 2):
            if i == 0:
                layers.append(nn.Linear(self.sizes[i], self.sizes[i + 1], bias=True))
            else:
                layers.append(nn.Linear(self.sizes[i], self.sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(self.sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(self.sizes[-2], self.sizes[-1], bias=False))
        return nn.Sequential(*layers).cuda()

