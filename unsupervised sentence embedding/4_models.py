import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import transformers
# Set PATHs
# import sys
# PATH_TO_MODEL = 'transformers-4.2.1/src'
#
# # Import SentEval
# sys.path.insert(0, PATH_TO_MODEL)
from modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
# from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers import BertTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class ProjectionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_dim = config.hidden_size
        hidden_dim = config.hidden_size * 2
        out_dim = config.hidden_size
        affine=False
        list_layers = [nn.Linear(in_dim, hidden_dim, bias=False),
                       nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(hidden_dim, out_dim, bias=False),
                        nn.BatchNorm1d(out_dim, affine=affine)]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x)


from scipy.sparse import block_diag
def simcse_unsup_loss(y_pred, triple=True, temp=0.05, loss1=True, extra_label=False, alpha=None):
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]
    """
    device = y_pred.device
    if triple:
        a = [[0, 0, 1], [0, 0, 1], [1, 1, 0]]
        y_true = block_diag([a for _ in range(y_pred.shape[0]//3)]).toarray()   # a是一个3*3的矩阵，然后扩充为对角线为a的3*64的矩阵，构造标签
        y_true = torch.Tensor(y_true).to(device)

    else:
        if not extra_label:
            a = [[0, 1], [1, 0]]
            y_true = block_diag([a for _ in range(y_pred.shape[0] // 2)]).toarray()
            y_true = torch.Tensor(y_true).to(device)
        else:
            a = [[0, 0.5], [0.5, 0]]
            y_true = block_diag([a for _ in range(y_pred.shape[0] // 2)]).toarray()
            y_true = torch.Tensor(y_true).to(device)
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    # y_pred_clone = y_pred.clone()

    # y_clone = y_pred[2::3].clone()
    # random_cls = y_clone.data.new(y_clone.size()).normal_()
    # y_pred = torch.cat([y_pred[::3], y_pred[1::3], random_cls], dim=1).view(-1, y_pred.shape[1])

    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)  # 相似度计算，形成一个相似度矩阵
    # sim_clone = sim.clone()
    if loss1:
        # sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
        # sim = sim / temp
        # filter_matrix = torch.tril(torch.triu(torch.ones_like(sim), 1), 25).bool()
        # filter_sim, filter_label = sim[filter_matrix], y_true[filter_matrix]
        # loss = cosent(filter_sim, filter_label)
        # loss2 = torch.mean(F.cross_entropy(sim, y_true))
        # loss = loss1 + loss2 * 0.1

        # sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12

        ##############################################################################################
        b = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
        filter_block = torch.tensor(block_diag([b for _ in range(sim.shape[0]//3)]).toarray())
        # filter_matrix = (torch.ones_like(filter_block) - filter_block).cuda()     # 构造一个过滤矩阵，把正样本对和与自己计算相似度的过滤掉

        # y_pred_ = torch.cat([y_pred[::3], y_pred[1::3]], dim=1).view(-1, y_pred.shape[1])
        # sim = F.cosine_similarity(y_pred_.unsqueeze(1), y_pred_.unsqueeze(0), dim=-1)
        # filter_sim = sim / temp
        # a = [[0, 1], [1, 0]]
        # y_true = block_diag([a for _ in range(y_pred_.shape[0] // 2)]).toarray()
        # y_true = torch.Tensor(y_true).to(device)

        filter_sim = sim - filter_block.cuda() * 1e12
        filter_sim = filter_sim / temp    # 除以温度系数
        # filter_sim = torch.tensor([[0.01, 0.01, 0.]])
        # y_true = torch.tensor([[0.5, 0.5, 0.]])

        loss_n = F.cross_entropy(filter_sim, y_true)

        ##############################################################################################

        # ones_matrix_row, ones_matrix_col = torch.zeros_like(sim), torch.zeros_like(sim)
        # ones_matrix_row[2::3], ones_matrix_col[:, 2::3] = 1, 1
        # ones_matrix = ones_matrix_row + ones_matrix_col
        # filter_matrix = torch.where(ones_matrix > 1.5, torch.zeros_like(sim).cuda(), filter_matrix.float())

        # filter_matrix = torch.tril(torch.triu(torch.tensor(filter_block), -2), 2).bool()
        # filter_further = torch.tril(torch.triu(torch.ones_like(sim), 1), 10)
        # final_filter = torch.where(filter_further == 1, filter_matrix, torch.zeros_like(sim).int()).bool()
        # filter_sim, filter_label = sim[final_filter], y_true[final_filter]
        # filter_sim, filter_label = sim[filter_matrix], y_true[filter_matrix]
        # loss_n = cosent(filter_sim, filter_label)
        # loss_n = F.binary_cross_entropy_with_logits(sim, y_true)

        # ones_matrix = torch.zeros_like(sim)
        # ones_matrix[2::3], ones_matrix[:, 2::3] = 1, 1
        # c = [[0, 1, 0], [1, 0, 0], [1, 1, 0]]
        # y_true_ori = block_diag([c for _ in range(y_pred.shape[0] // 3)]).toarray()
        # y_true_ori = torch.Tensor(y_true_ori).to(device)

        # sim = F.cosine_similarity(y_pred[::3].unsqueeze(1), y_pred[1::3].unsqueeze(0), dim=-1)
        # labels = torch.arange(sim.size(0)).long().cuda()
        # y_true = make_one_hot(labels, labels.shape[0])
        # sim = sim / temp
        # ones = torch.ones_like(sim)
        # filter_matrix = torch.tril(torch.triu(ones, -30), 25).bool()
        # filter_sim, filter_label = sim[filter_matrix], y_true[filter_matrix]
        # loss_o = cosent(filter_sim, filter_label)

        y_pred_ = torch.cat([y_pred[::3], y_pred[1::3]], dim=1).view(-1, y_pred.shape[1])
        sim = F.cosine_similarity(y_pred_.unsqueeze(1), y_pred_.unsqueeze(0), dim=-1)
        sim = sim / temp
        a = [[0, 1], [1, 0]]
        y_true = block_diag([a for _ in range(y_pred_.shape[0] // 2)]).toarray()
        y_true = torch.Tensor(y_true).to(device)

        # eye = torch.eye(sim.shape[0], device=device)
        # # filter_m = torch.where(ones_matrix == 1, ones_matrix, eye)
        # sim = sim - eye * 1e12
        #
        # loss_o = F.cross_entropy(sim, y_true)

        b = torch.eye(sim.shape[0]).cuda()
        c = torch.ones_like(sim) - b
        filter_matrix = torch.tril(torch.triu(c, 1), 25).bool()
        filter_sim, filter_label = sim[filter_matrix], y_true[filter_matrix]
        loss_o = cosent(filter_sim, filter_label)

        # alpha = 0.12
        loss = loss_n * alpha + loss_o  #* (1 - alpha)
    else:
        sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
        sim = sim / temp
        filter_matrix = torch.tril(torch.triu(torch.ones_like(sim), 1), 31).bool()
        filter_sim, filter_label = sim[filter_matrix].unsqueeze(0), y_true[filter_matrix].unsqueeze(0)
        # filter_sim = torch.where(filter_matrix < 1, sim - 1e12, sim)
        loss1 = F.cross_entropy(filter_sim, filter_label)
        loss2 = torch.mean(F.cross_entropy(sim, y_true))
        loss = loss2 + loss1 * 0.01
        # a = torch.logsumexp(torch.log(torch.tensor(sim.shape[0]).cuda()), dim=0)
        # loss = torch.mean(F.cross_entropy(sim, y_true) - a)
        # n = torch.log(sim.shape[0] - torch.tensor([1]).float()).cuda()
        # y_pred = torch.cat((n, sim), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
        # loss = torch.logsumexp(y_pred, dim=0) - torch.logsumexp(n, dim=0)
        # loss = torch.mean(F.nll_loss(sim, y_true))
    return loss


def make_one_hot(labels, classes):
    # onehot = torch.FloatTensor(labels.shape[0], classes, labels.shape[1]).zero_().cuda()
    onehot = torch.zeros((labels.shape[0], classes)).cuda()
    target = onehot.scatter_(1, labels.unsqueeze(1), 1)
    target[:, 0] = 0
    return target


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


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        self.record = None
        self.pos_avg = 0.0
        self.neg_avg = 0.0

    def forward(self, x, y):
        sim = self.cos(x, y)
        self.record = sim.detach()
        min_size = min(self.record.shape[0], self.record.shape[1])
        num_item = self.record.shape[0] * self.record.shape[1]
        self.pos_avg = self.record.diag().sum() / min_size
        self.neg_avg = (self.record.sum() - self.record.diag().sum()) / (num_item - min_size)
        return sim / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config) if not cls.model_args.batchnorm else ProjectionMLP(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()
    # cls.generator = transformers.DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased') if cls.model_args.generator_name is None else transformers.AutoModelForMaskedLM.from_pretrained(cls.model_args.generator_name)
    cls.electra_acc = 0.0
    cls.electra_rep_acc = 0.0
    cls.electra_fix_acc = 0.0


def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    cls_token=101,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)

    # predict = nn.Linear(768, 30524).cuda()
    # output_id = cls.lm_head(cls.mlp(pooler_output))
    # seq_output = torch.max(output_id, 2)[1]
    # tokenizer = BertTokenizer.from_pretrained('../Law/pretrained_model/bert_based_uncased_english')
    #
    # fw = open('return_sequence4.txt', 'a', encoding='utf-8')
    # for i in range(seq_output.shape[0]):
    #     sequence = tokenizer.convert_ids_to_tokens(seq_output[i])
    #     ori_sequence = tokenizer.convert_ids_to_tokens(input_ids[i])
    #     sequence = tokenizer.convert_tokens_to_string(sequence)
    #     ori_sequence = tokenizer.convert_tokens_to_string(ori_sequence)
    #     show = f'{sequence}\n{ori_sequence}\n'
    #     fw.write(show)

    # pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)
    
    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if not cls.model_args.before_mlp:
        if cls.pooler_type == "cls":
            pooler_output = pooler_output.view((batch_size*num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)
            pooler_output = cls.mlp(pooler_output)
            pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)
    # Produce MLM augmentations and perform conditional ELECTRA using the discriminator
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        with torch.no_grad():
            g_pred = cls.generator(mlm_input_ids, attention_mask)[0].argmax(-1)
        g_pred[:, 0] = cls_token
        replaced = (g_pred != input_ids) * attention_mask
        e_inputs = g_pred * attention_mask

        mlm_outputs = cls.discriminator(
            e_inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
            cls_input=pooler_output.view((-1, pooler_output.size(-1))),
        )

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.model_args.before_mlp:
        if cls.pooler_type == "cls":
            pooler_output = cls.mlp(pooler_output)

    # Separate representation
    # z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # # Hard negative
    # if num_sent == 3:
    #     z3 = pooler_output[:, 2]
    #
    # # Gather all embeddings if using distributed training
    # if dist.is_initialized() and cls.training:
    #     # Gather hard negative
    #     if num_sent >= 3:
    #         z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
    #         dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
    #         z3_list[dist.get_rank()] = z3
    #         z3 = torch.cat(z3_list, 0)
    #
    #     # Dummy vectors for allgather
    #     z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
    #     z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
    #     # Allgather
    #     dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
    #     dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
    #
    #     # Since allgather results do not have gradients, we replace the
    #     # current process's corresponding embeddings with original tensors
    #     z1_list[dist.get_rank()] = z1
    #     z2_list[dist.get_rank()] = z2
    #     # Get full batch embeddings: (bs x N, hidden)
    #     z1 = torch.cat(z1_list, 0)
    #     z2 = torch.cat(z2_list, 0)

    # cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    # if num_sent >= 3:
    #     z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
    #     cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    # labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    # loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    # if num_sent == 3:
    #     # Note that weights are actually logits of weights
    #     z3_weight = cls.model_args.hard_negative_weight
    #     weights = torch.tensor(
    #         [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
    #     ).to(cls.device)
    #     cos_sim = cos_sim + weights
    #
    # loss = loss_fct(cos_sim, labels)

    # Calculate loss for conditional ELECTRA
    cos_sim = 0
    loss = simcse_unsup_loss(y_pred=pooler_output, triple=cls.model_args.multi_dropout, loss1=True, alpha=cls.model_args.alpha)
    if mlm_outputs is not None and mlm_labels is not None:
        # mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        e_labels = replaced.view(-1, replaced.size(-1))
        # prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        prediction_scores = cls.electra_head(mlm_outputs.last_hidden_state)
        rep = (e_labels == 1) * attention_mask
        fix = (e_labels == 0) * attention_mask
        prediction = prediction_scores.argmax(-1)
        cls.electra_rep_acc = float((prediction*rep).sum()/rep.sum())
        cls.electra_fix_acc = float(1.0 - (prediction*fix).sum()/fix.sum())
        cls.electra_acc = float(((prediction == e_labels) * attention_mask).sum()/attention_mask.sum())
        # masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        masked_lm_loss = loss_fct(prediction_scores.view(-1, 2), e_labels.view(-1))
        loss = loss + cls.model_args.lambda_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, args):
        super().__init__(config)
        # self.model_args = model_kargs["model_args"]
        self.model_args = args
        self.bert = BertModel(config, add_pooling_layer=False)

        # self.classifier = nn.Linear(config.hidden_size, 1, bias=True)
        # self.lm_head = BertLMPredictionHead(config)
        # self.discriminator = BertModel(config, add_pooling_layer=False)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        sent_emb=True,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                cls_token=101,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, args):
        super().__init__(config)
        # self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.model_args = args
        # self.classifier = nn.Linear(config.hidden_size, 1, bias=True)
        # self.lm_head = RobertaLMHead(config)
        # self.discriminator = RobertaModel(config, add_pooling_layer=False)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=True,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                cls_token=0,
            )
