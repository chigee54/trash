import datetime
import json, os, math
from os.path import join
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import logging
from torch.nn import init
import torch.nn.functional as F


def load_checkpoint(model, optimizer, trained_epoch):
    filename = args.output_path + '/' + f"extract-{trained_epoch}.pkl"
    save_params = torch.load(filename)
    model.load_state_dict(save_params["model"])
    # optimizer.load_state_dict(save_params["optimizer"])


def save_checkpoint(model, optimizer, trained_epoch):
    # save_params = {
    #     "model": model.state_dict(),
    #     "optimizer": optimizer.state_dict(),
    #     "trained_epoch": trained_epoch,
    # }
    if not os.path.exists(args.output_path):
        # 判断文件夹是否存在，不存在则创建文件夹
        os.mkdir(args.output_path)
    filename = args.output_path + '/' + f"extract-{trained_epoch}.pkl"
    torch.save(model, filename)


def load_bm25_scores(scores_file, data_file, embedding_npy):
    scores = np.zeros_like(embedding_npy[..., :1])
    with open(data_file, encoding='utf-8') as f:
        i, j = 0, 0
        for line in json.load(f):
            ids = line['labels']
            scores[i, j] = scores_file[str(ids[0])][str(ids[1])]
            if j < 99:
                j += 1
            else:
                i += 1
                j = 0
    return scores


def load_labels(filename, embedding_npy, type_='train'):
    """加载标签
    """
    labels = np.zeros_like(embedding_npy[..., :3]) if type_ == 'train' else np.zeros_like(embedding_npy[..., :2])
    with open(filename, encoding='utf-8') as f:
        i, j = 0, 0
        for line in json.load(f):
            if len(line['labels']) < 3:
                line['labels'] = [1, 1, 1]
            labels[i, j] = line['labels']
            if j < 99:
                j += 1
            else:
                i += 1
                j = 0
    return labels


class SimplifiedScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''
    def __init__(self, d_model, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SimplifiedScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model//h
        self.d_v = d_model//h
        self.h = h
        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout=nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class ResidualGatedConv1D(nn.Module):
    """门控卷积
    """
    def __init__(self, filters, kernel_size, dilation_rate=1):
        super(ResidualGatedConv1D, self).__init__()
        self.filters = filters  # 输出维度
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.supports_masking = True
        self.padding = self.dilation_rate*(self.kernel_size - 1)//2
        self.conv1d = nn.Conv1d(filters, 2*filters, self.kernel_size, padding=self.padding, dilation=self.dilation_rate)
        self.layernorm = nn.LayerNorm(self.filters)
        # self.batchnorm = BatchNormMLP(self.filters)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        input_cov1d = inputs.permute([0, 2, 1])
        outputs = self.conv1d(input_cov1d)
        outputs = outputs.permute([0, 2, 1])
        gate = torch.sigmoid(outputs[..., self.filters:])
        outputs = outputs[..., :self.filters] * gate
        # outputs = self.batchnorm(outputs.squeeze()).unsqueeze(0)
        outputs = self.layernorm(outputs)

        if hasattr(self, 'dense'):
            inputs = self.dense(inputs)

        return inputs + self.alpha * outputs


# class Regressor(nn.Sequential):
#     def __init__(self, input_channel, output_channel):
#         super(Regressor, self).__init__()
#         self.convA = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1)
#         self.leakyreluA = nn.ReLU()
#         self.convB = nn.Conv2d(output_channel, output_channel, kernel_size=1, stride=1)
#         self.leakyreluB = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.5)
#         self.convC = nn.Conv2d(output_channel, 1, kernel_size=1, stride=1)
#         self.activation = nn.Tanh()
#
#     def forward(self, x):
#         x = self.convA(x)
#         x = self.leakyreluA(x)
#         x = self.convB(x)
#         x = self.leakyreluB(x)
#         x = self.dropout(x)
#         x = self.convC(x)
#
#         return self.activation(x)


class BatchNormMLP(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        in_dim = hidden_size
        hidden_dim = hidden_size * 2
        out_dim = hidden_size
        affine = False
        list_layers = [nn.Linear(in_dim, hidden_dim, bias=False),
                       nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(hidden_dim, out_dim, bias=False),
                        nn.BatchNorm1d(out_dim, affine=affine)]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x)


class Selector2(nn.Module):
    def __init__(self, input_size, filters, kernel_size, dilation_rate):
        """
        :param feature_size:每个词向量的长度
        """
        super(Selector2, self).__init__()
        list_layers_1 = [nn.Dropout(0.1),
                         nn.Linear(input_size, filters, bias=False),
                         ResidualGatedConv1D(filters, kernel_size, dilation_rate=dilation_rate[0]),
                         nn.SELU(),
                         # nn.BatchNorm1d(200, affine=False),
                         # nn.Dropout(0.05),
                         # nn.Linear(filters, 1,)
        ]
        self.net1 = nn.Sequential(*list_layers_1)

    def forward(self, inputs, labels=None, ):
        inputs_sep = [inputs[..., :768], inputs[..., 768:]]
        double_inputs = torch.cat([inputs_sep[1], inputs_sep[1]], dim=2)
        double_inputs = double_inputs.view(1, -1, 768)
        output = self.net1(double_inputs)
        return output, labels


class Selector_Dataset(Dataset):
    def __init__(self, data_x, data_y):
        super(Selector_Dataset, self).__init__()
        self.data_x_tensor = torch.from_numpy(data_x)
        # self.score_x_tensor = torch.from_numpy(score_x)
        self.data_y_tensor = torch.from_numpy(data_y)

    def __len__(self):
        return len(self.data_x_tensor)

    def __getitem__(self, idx):
        return self.data_x_tensor[idx], self.data_y_tensor[idx]


from scipy.sparse import block_diag
def cosent_loss(y_pred, temp=20):
    a = [[0, 1], [1, 0]]
    y_true = block_diag([a for _ in range(y_pred.shape[0] // 2)]).toarray()
    y_true = torch.Tensor(y_true).to(device)
    y_pred = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    b = torch.eye(y_pred.shape[0]).to(device)
    c = torch.ones_like(y_pred) - b
    filter_matrix = torch.tril(torch.triu(c, 1), 25).bool()
    y_pred, y_true = y_pred[filter_matrix], y_true[filter_matrix]
    # y_true = y_true.view(-1)
    # y_pred = y_pred.view(-1)
    # pn_index = (y_true > 0).long()
    y_pred = y_pred * temp  # * (pn_index * 0.5)
    y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置 两两之间余弦的差值
    y_true = y_true[:, None] < y_true[None, :]  # 取出负例-正例的差值
    y_true = y_true.float()
    y_pred = y_pred - (1 - y_true) * 1e12
    y_pred = y_pred.view(-1)
    y_pred = torch.cat((torch.tensor([0]).float().to(device), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
    return torch.logsumexp(y_pred, dim=0)


def ndcg(ranks, K):
    dcg_value = 0.
    idcg_value = 0.

    sranks = sorted(ranks, reverse=True)

    for i in range(0, K):
        logi = math.log(i+2, 2)
        dcg_value += ranks[i] / logi
        idcg_value += sranks[i] / logi
    if idcg_value == 0.0:
        idcg_value += 0.00000001
    return dcg_value/idcg_value


def cal_ndcg(all_preds, all_labels):
    ndcgs = []
    for qidx, pred_ids in all_preds.items():
        did2rel = all_labels[str(qidx)]
        ranks = [did2rel[str(idx)] if str(idx) in did2rel else 0 for idx in pred_ids]
        ndcgs.append(ndcg(ranks, 30))
        # print(f'********** qidx: {qidx} **********')
        # print(f'top30 pred_ids: {pred_ids}')
        # print(f'ranks: {ranks}')
    # print(ndcgs)
    return sum(ndcgs) / len(ndcgs)


def evaluate(model, dataloader, all_labels=None):
    model.eval()
    with torch.no_grad():
        all_preds = {}
        for data in dataloader:
            x_batch, labels = data
            x_batch = x_batch.to(device)
            # x_scores = x_scores.to(device)
            scores, _ = model(x_batch, )  #
            labels = labels.squeeze().cpu().tolist()
            scores = scores.squeeze().cpu().tolist()
            preds, dids, qidx = [], [], []
            for label, score in zip(labels, scores):
                preds.append(score)
                dids.append(int(label[1]))
                qidx.append(int(label[0]))
            sorted_r = sorted(list(zip(dids, preds)), key=lambda x: x[1], reverse=True)
            pred_ids = [x[0] for x in sorted_r]
            assert len(set(qidx)) == 1
            all_preds[qidx[0]] = pred_ids[:30]
    ndcg_30 = cal_ndcg(all_preds, all_labels)
    # if ndcg_30 > 0.949:
    #     test_extract_json = './data/phase_2/test_data_win200.json'
    #     test_extract_npy = './data/phase_2/test_embeddings_win200.npy'
    #     test_x = np.load(test_extract_npy)
    #     test_y = load_labels(test_extract_json, test_x, 'test')
    #     test_dataloader = DataLoader(Selector_Dataset(test_x, test_y), batch_size=1, shuffle=False)
    #     test_result(model, test_dataloader, args)
    torch.cuda.empty_cache()
    return ndcg_30


def train(model, train_dataloader, valid_dataloader):
    all_labels = json.load(open(args.label_file, 'r', encoding='utf8'))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # criterion = nn.BCELoss(reduction='none')
    best = 100
    for epoch in tqdm(range(args.epoch_num)):
        epoch_loss = 0.0
        current_step = 0
        model.train()
        for batch_data in train_dataloader:
            x_batch, label_batch = batch_data
            x_batch = x_batch.to(device)
            # x_scores = x_scores.to(device)
            label_batch = label_batch[..., 2].to(device)
            output_batch, _ = model(x_batch)
            # output_batch = output_batch.permute([0, 2, 1])
            loss = cosent_loss(output_batch.squeeze(), )
            # loss = ndcg_listwise(output_batch.squeeze(), label_batch.squeeze().to(device))
            # loss = torch.div(torch.sum(loss*batch_mask), torch.sum(batch_mask))
            optimizer.zero_grad()
            # loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            loss_item = loss.cpu().detach().item()
            epoch_loss += loss_item
            current_step += 1

        epoch_loss = epoch_loss / current_step
        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # print('{} train epoch {} loss: {:.4f}'.format(time_str, epoch, epoch_loss))
        # load_checkpoint(model, optimizer, 'best')
        # model = torch.load(args.output_path + '/' + f"extract-best.pkl")['model']
        # ndcg30 = evaluate(model, valid_dataloader, all_labels)
        # print(ndcg30)
        # exit(0)
        # ndcg30 = evaluate(model, valid_dataloader, all_labels)
        # if epoch+1 in [299, 659, 991]:
        print('Epoch[{}/{}], loss:{}'.format(epoch + 1, args.epoch_num, epoch_loss))
        if best > epoch_loss:
        # if epoch_loss < 5.0:
            best = epoch_loss
            save_checkpoint(model, optimizer, 'best')
            print('loss: {}, Epoch[{}/{}], loss:{}, save model\n'.format(best, epoch + 1, args.epoch_num, epoch_loss))
            logging.info('loss: {}, Epoch[{}/{}], loss:{}, save model\n'.format(best, epoch + 1, args.epoch_num,
                                                                                  epoch_loss))
            if best < 5:
                exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--epoch_num', type=int, default=5000, help='number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='decay weight of optimizer')
    parser.add_argument('--output_path', type=str, default="./result/3", help='checkpoint path')
    parser.add_argument('--input_size', type=int, default=768)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--dilation_rate', type=list, default=[1, 2, 4, 8, 8, 1])
    parser.add_argument('--label_file', type=str, default='./data/phase_1/train/label_top30_dict.json')
    parser.add_argument('--train_label_file', type=str, default='./data/phase_2/train/label_top30_dict.json')
    parser.add_argument('--cuda_pos', type=str, default='1', help='which GPU to use')
    parser.add_argument('--seed', type=int, default=42, help='max length of each case')
    args = parser.parse_args()
    # config environment
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    args.output_path = join(args.output_path, 'seed{}-bsz{}-lr{}'.format(args.seed, args.batch_size, args.lr))
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    log_name = "log_DGCNN"
    logging.basicConfig(level=logging.INFO,  # 控制台打印的日志级别
                        filename=args.output_path + '/{}.log'.format(log_name),
                        filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        # a是追加模式，默认如果不写的话，就是追加模式
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        # 日志格式
                        )

    # 配置信息
    device = torch.device('cuda:' + args.cuda_pos) if torch.cuda.is_available() else torch.device('cpu')

    # training
    dev_extract_json = './data/phase_2/dev_data_win200.json'
    dev_extract_npy = './data/phase_2/dev_embeddings_win200.npy'
    train_extract_json = './data/phase_2/train_data_win200.json'
    train_extract_npy = './data/phase_2/train_embeddings_win200.npy'

    test_extract_json = './data/phase_2/test_data_win200.json'
    test_extract_npy = './data/phase_2/test_embeddings_pro.npy'
    test_x = np.load(test_extract_npy)
    test_y = load_labels(test_extract_json, test_x)

    dev_x = np.load(dev_extract_npy)
    train_x = np.load(train_extract_npy)
    # bm25_scores_dev_file = json.load(open('result/bm25_scores_dev.json', 'r', encoding='utf8'))
    # bm25_scores_dev = load_bm25_scores(bm25_scores_dev_file, dev_extract_json, dev_x)
    # bm25_scores_train_file = json.load(open('result/bm25_scores_train.json', 'r', encoding='utf8'))
    # bm25_scores_train = load_bm25_scores(bm25_scores_train_file, train_extract_json, train_x)
    dev_y = load_labels(dev_extract_json, dev_x)
    train_y = load_labels(train_extract_json, train_x)

    train_y = np.concatenate([train_y, dev_y, test_y], axis=0)
    train_x = np.concatenate([train_x, dev_x, test_x], axis=0)

    train_dataloader = DataLoader(Selector_Dataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(Selector_Dataset(dev_x, dev_y), batch_size=1, shuffle=False)

    model = Selector2(args.input_size, args.hidden_size, kernel_size=args.kernel_size, dilation_rate=args.dilation_rate)
    # load_checkpoint(model, None, '0.9455')
    # model.to(device)
    # ndcg30 = evaluate(model, valid_dataloader, json.load(open(args.label_file, 'r', encoding='utf8')))

    train(model, train_dataloader, valid_dataloader)

