import torch
import torch.nn as nn
from transformers import AutoConfig
from modeling_bert import BertModel
import torch.nn.functional as F


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

    def __init__(self, config, pre_seq_len=10, prefix_projection=False):
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


class Custom_Model(nn.Module):
    """模型自定义"""
    def __init__(self, pretrained_model, pooling='cls', pre_seq_len=10, num_labels=1, dropout=0.1, multi_dropout=False,
                 dropout_noise=None, do_ptuning=False, prefix_projection=False, classification=False):
        super(Custom_Model, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model)
        if multi_dropout is not None:
            config.multi_dropout = multi_dropout
            config.attention_probs_dropout_prob_noise = dropout_noise
            config.hidden_dropout_prob_noise = dropout_noise
        config.attention_probs_dropout_prob = dropout
        config.hidden_dropout_prob = dropout
        self.classification = classification
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.mlp = MLPLayer(config)
        self.pooling = pooling
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)   # the num_labels of STS-B is 1
        self.do_ptuning = do_ptuning

        ##########################################################################
        if self.do_ptuning:
            self.pre_seq_len = pre_seq_len
            self.n_layer = config.num_hidden_layers
            self.n_head = config.num_attention_heads
            self.n_embd = config.hidden_size // config.num_attention_heads

            self.prefix_tokens = torch.arange(pre_seq_len).long()
            self.prefix_encoder = PrefixEncoder(config, pre_seq_len=pre_seq_len, prefix_projection=prefix_projection)
            self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.prefix_encoder.parameters():
                param.requires_grad = True

            # compute the number of total parameters and tunable parameters
            total_param = sum(p.numel() for p in self.parameters())
            trainable_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print('total param is {}, trainable param is {}, Rate is {}'.format(total_param, trainable_param, trainable_param/total_param))
        ##########################################################################

    def forward(self, input_ids, attention_mask, token_type_ids, mlp_cls=True, past_key_values=None):

        ##########################################################################
        if self.do_ptuning:
            # concatenate prefix_key and prefix_value with original key and value in hidden states
            past_key_values = self.get_prompt(batch_size=input_ids.shape[0])
            prefix_attention_mask = torch.ones(input_ids.shape[0], self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        ##########################################################################
        # [last_hidden_state, pooler_output, hidden_states, attentions]
        out = self.bert(input_ids, attention_mask, token_type_ids, past_key_values=past_key_values)
        if self.classification:
            pooled_output = self.dropout(out[1])
            logits = self.classifier(pooled_output)
            return (logits,) + out[2:]  # add hidden states and attention if they are here
        if self.pooling == 'cls':
            # out[0] is last_hidden_state
            return self.mlp(out.last_hidden_state[:, 0]) if mlp_cls is True else out.last_hidden_state[:, 0]
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
        # change into the format needed for bert
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(batch_size, self.pre_seq_len, self.n_layer * 2, self.n_head, self.n_embd)
        past_key_values = self.dropout(past_key_values)
        # transpose dimension
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
    ##########################################################################


def simcse_unsup_loss(y_pred, device, temp=0.05):
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]

    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / temp
    # 计算相似度矩阵与y_true的交叉熵损失
    # 计算交叉熵，每个case都会计算与其他case的相似度得分，得到一个得分向量，目的是使得该得分向量中正样本的得分最高，负样本的得分最低
    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)


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


def cosent_unsup_loss(y_pred, y_true, temp=20):
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

