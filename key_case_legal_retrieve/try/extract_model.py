import datetime
import json, os, sys
from os.path import join
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import logging


def load_checkpoint(model, optimizer, trained_epoch):
    filename = args.output_path + '/' + f"extract-{trained_epoch}.pkl"
    save_params = torch.load(filename)
    model.load_state_dict(save_params["model"])
    #optimizer.load_state_dict(save_params["optimizer"])


def save_checkpoint(model, optimizer, trained_epoch):
    save_params = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
    }
    if not os.path.exists(args.output_path):
        # 判断文件夹是否存在，不存在则创建文件夹
        os.mkdir(args.output_path)
    filename = args.output_path + '/' + f"extract-{trained_epoch}.pkl"
    torch.save(save_params, filename)


def load_labels(filename, embedding_npy):
    """加载数据
    返回：[(texts, labels, summary)]
    """
    labels = np.zeros_like(embedding_npy[..., :1])
    with open(filename, encoding='utf-8') as f:
        for i, line in enumerate(json.load(f)):
            for j in line['labels']:
                labels[i, j] = 1
    return labels


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
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        input_cov1d = inputs.permute([0, 2, 1])
        outputs = self.conv1d(input_cov1d)
        outputs = outputs.permute([0, 2, 1])
        gate = torch.sigmoid(outputs[..., self.filters:])
        outputs = outputs[..., :self.filters] * gate
        outputs = self.layernorm(outputs)

        if hasattr(self, 'dense'):
            inputs = self.dense(inputs)

        return inputs + self.alpha * outputs


class Selector2(nn.Module):
    def __init__(self, input_size, filters, kernel_size, dilation_rate):
        """
        :param feature_size:每个词向量的长度
        """
        super(Selector2, self).__init__()
        self.dense1 = nn.Linear(input_size, filters, bias=False)
        self.ResidualGatedConv1D_1 = ResidualGatedConv1D(filters, kernel_size, dilation_rate=dilation_rate[0])
        self.ResidualGatedConv1D_2 = ResidualGatedConv1D(filters, kernel_size, dilation_rate=dilation_rate[1])
        self.ResidualGatedConv1D_3 = ResidualGatedConv1D(filters, kernel_size, dilation_rate=dilation_rate[2])
        self.ResidualGatedConv1D_4 = ResidualGatedConv1D(filters, kernel_size, dilation_rate=dilation_rate[3])
        self.ResidualGatedConv1D_5 = ResidualGatedConv1D(filters, kernel_size, dilation_rate=dilation_rate[4])
        self.ResidualGatedConv1D_6 = ResidualGatedConv1D(filters, kernel_size, dilation_rate=dilation_rate[5])
        self.dense2 = nn.Linear(filters, 1)

    def forward(self, inputs):
        mask = inputs.ge(0.00001)
        mask = torch.sum(mask, axis=-1).bool()
        x1 = self.dense1(nn.Dropout(0.1)(inputs))
        x2 = self.ResidualGatedConv1D_1(nn.Dropout(0.1)(x1))
        x3 = self.ResidualGatedConv1D_2(nn.Dropout(0.1)(x2))
        x4 = self.ResidualGatedConv1D_3(nn.Dropout(0.1)(x3))
        x5 = self.ResidualGatedConv1D_4(nn.Dropout(0.1)(x4))
        x6 = self.ResidualGatedConv1D_5(nn.Dropout(0.1)(x5))
        x7 = self.ResidualGatedConv1D_6(nn.Dropout(0.1)(x6))
        output = nn.Sigmoid()(self.dense2(nn.Dropout(0.1)(x7)))
        return output, mask


class Selector_Dataset(Dataset):
    def __init__(self, data_x, data_y):
        super(Selector_Dataset, self).__init__()
        self.data_x_tensor = torch.from_numpy(data_x)
        self.data_y_tensor = torch.from_numpy(data_y)

    def __len__(self):
        return len(self.data_x_tensor)

    def __getitem__(self, idx):
        return self.data_x_tensor[idx], self.data_y_tensor[idx]


def train(model, train_dataloader, valid_dataloader):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss(reduction='none')
    lower_loss = 1.0
    higher_recall = 0.0
    for epoch in range(args.epoch_num):
        epoch_loss = 0.0
        current_step = 0
        model.train()
        pbar = tqdm(train_dataloader, desc="Iteration", postfix='train')
        for batch_data in pbar:
            x_batch, label_batch = batch_data
            x_batch = x_batch.to(device)
            label_batch = label_batch.to(device)
            output_batch, batch_mask = model(x_batch)
            output_batch = output_batch.permute([0, 2, 1])
            loss = criterion(output_batch.squeeze(), label_batch.squeeze())
            loss = torch.div(torch.sum(loss*batch_mask), torch.sum(batch_mask))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_item = loss.cpu().detach().item()
            epoch_loss += loss_item
            current_step += 1
            pbar.set_description("train loss {}".format(epoch_loss / current_step))
            if current_step % 100 == 0:
                logging.info("train step {} loss {}".format(current_step, epoch_loss / current_step))

        epoch_loss = epoch_loss / current_step
        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # print('{} train epoch {} loss: {:.4f}'.format(time_str, epoch, epoch_loss))
        logging.info('train epoch {} loss: {:.4f}'.format(epoch, epoch_loss))
        if lower_loss > epoch_loss:
            print('{} train epoch {} lower loss: {:.4f}'.format(time_str, epoch, epoch_loss))
            logging.info('{} train epoch {} lower loss: {:.4f}'.format(time_str, epoch, epoch_loss))
            save_checkpoint(model, optimizer, epoch)
            lower_loss = epoch_loss
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            current_step = 0
            pbar = tqdm(valid_dataloader, desc="Iteration", postfix='valid')
            for batch_data in pbar:
                x_batch, label_batch = batch_data
                x_batch = x_batch.to(device)
                label_batch = label_batch.to(device).long()
                output_batch, batch_mask = model(x_batch)
                label_batch = label_batch.to(device)
                total += torch.sum(batch_mask)
                c = (output_batch.squeeze() > args.threshold).long()
                d = label_batch.squeeze().long()
                vec_correct = (c == d) * batch_mask
                # a = sum((c * batch_mask).reshape(-1))
                # b = sum((d * batch_mask).reshape(-1))
                # recall = a / b
                # if recall > higher_recall:
                #     print('{} train epoch {} higher recall: {:.4f}'.format(time_str, epoch, recall))
                #     logging.info('{} train epoch {} higher recall {:.4f}'.format(time_str, epoch, recall))
                #     save_checkpoint(model, optimizer, epoch)
                #     higher_recall = recall
                correct += torch.sum(vec_correct).cpu().item()
                pbar.set_description("valid acc {}".format(correct / total))
                current_step += 1
                if current_step % 100 == 0:
                    logging.info('valid epoch {}, acc {}/{}={:.4f}'.format(epoch, correct, total, correct / total))
            time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print('{} valid epoch {} acc {}/{}={:.4f}'.format(time_str, epoch, correct, total, correct / total))
            logging.info('valid epoch {} acc {}/{}={:.4f}'.format(epoch, correct, total, correct / total))


def query_convert(data_type):
    save_case = './data/query_convert/{}_query_cases.json'.format(data_type)
    query_data = json.load(open('data/query_convert/{}_query_extract.json'.format(data_type), 'r', encoding='utf8'))
    valid_x = np.load('data/query_convert/{}_query_extract.npy'.format(data_type))
    with torch.no_grad():
        model = Selector2(args.input_size, args.hidden_size, kernel_size=args.kernel_size, dilation_rate=[1, 2, 4, 8, 1, 1])
        load_checkpoint(model, None, 9)
        model_output = model(torch.tensor(valid_x))[0]
        y_pred = model_output.cpu().numpy()
    query_cases = []
    for d, yp in tqdm(zip(query_data, y_pred), desc=u'转换中'):
        if len(d['query']) < 1:
            continue
        yp = yp[:len(d['query'])]
        yp = np.where(yp > 0.1)[0]
        if len(yp) < 4:
            random_list = [i for i in range(2, 8) if i not in yp]
            yp = np.concatenate((yp, random_list[:4]), axis=-1)
        try:
            pre_case = '；'.join([d['query'][i] for i in yp])
        except:
            pre_case = '；'.join(d['query'])
        query_cases.append({'qidx': d['qidx'], 'query_case': pre_case})
    with open(save_case, 'w', encoding='utf8') as fs:
        json.dump(query_cases, fs, ensure_ascii=False, indent=2)


def new_query_and_candidate():
    save_file = 'data/test/case_test.json'
    fc = json.load(open('data/test/key_test.json', 'r', encoding='utf8'))
    fq = json.load(open('data/test/test_query_cases.json', 'r', encoding='utf8'))
    update = []
    for case in fq:
        for line in fc:
            if case['qidx'] == line['label'][0]:
                line['query'] = case['query_case']
                # ori_label = line['label'][2] * 3
                # if ori_label > 1:
                #     line['label'][2] = int(ori_label - 2)
                # else:
                #     line['label'][2] = int(ori_label)
                update.append(line)
    with open(save_file, 'w', encoding='utf8') as fs:
        json.dump(update, fs, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # new_query_and_candidate()
    # sys.exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--epoch_num', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='decay weight of optimizer')
    parser.add_argument('--output_path', type=str, default="./result/", help='checkpoint path')
    parser.add_argument('--input_size', type=int, default=768)
    parser.add_argument('--hidden_size', type=int, default=384)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--cuda_pos', type=str, default='1', help='which GPU to use')
    parser.add_argument('--seed', type=int, default=42, help='max length of each case')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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

    query_convert('small')
    sys.exit(0)
    # 加载数据
    dev_extract_json = './data/extract_data/extract_small.json'
    dev_extract_npy = './data/extract_data/small_extract.npy'
    train_extract_json = './data/extract_data/extract_all.json'
    train_extract_npy = './data/extract_data/large_extract_all.npy'

    dev_x = np.load(dev_extract_npy)
    train_x = np.load(train_extract_npy)
    dev_y = load_labels(dev_extract_json, dev_x)
    train_y = load_labels(train_extract_json, train_x)

    train_dataloader = DataLoader(Selector_Dataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(Selector_Dataset(dev_x, dev_y), batch_size=len(dev_x), shuffle=False)

    model = Selector2(args.input_size, args.hidden_size, kernel_size=args.kernel_size, dilation_rate=[1, 2, 4, 8, 1, 1])

    train(model, train_dataloader, valid_dataloader)

    # query_convert()

