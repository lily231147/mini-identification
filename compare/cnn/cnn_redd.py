import torch
from torch import nn, optim
from torch.utils import data as data_
import os
import numpy as np

train_datapath = 'E:/NILM/CNN_Compare/'
LABEL_NAMES = [0, 10, 14, 16]
train_mode = 'test'


# 模型，最开始的输入是(1,9)
class cnn_net(nn.Module):
    def __init__(self, n_class, length):
        super(cnn_net, self).__init__()
        self.n_class = n_class
        self.length = length
        # there is a max_pooling followed by relu, but we remove it for the reasons:
        # 1) there is no need for pooling layer as the input only has 7 points
        # 2) we don't know the params of pooling
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 10, 3, 1, 1),  # input 1,L
            nn.BatchNorm1d(10),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(10, 20, 3, 1, 1),  # input 10,L
            nn.BatchNorm1d(20),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(20, 30, 3, 1, 1),  # input 20,L
            nn.BatchNorm1d(30),
            nn.ReLU()  # output 30,9
        )
        self.fc1 = nn.Sequential(
            nn.Linear(30 * self.length, 60),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(60, 15),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(15, self.n_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# 选择模型
model = cnn_net(n_class=4)
if os.path.exists(train_datapath + 'parameter.pkl'):
    model.load_state_dict(torch.load(train_datapath + 'parameter.pkl'))
else:
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight.data)
            nn.init.xavier_normal_(m.weight.data)
            nn.init.kaiming_normal_(m.weight.data)  # 卷积层参数初始化
            m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_()  # 全连接层参数初始化
model = model.cuda()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)


class VOCBboxDataset:
    def __init__(self, data_dir, split=train_mode):
        id_list_file = os.path.join(data_dir, '{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.datas = []
        for id in self.ids:
            self.datas.append(np.loadtxt(self.data_dir + str(id) + '.txt'))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        data_original = self.datas[i]
        img = data_original[:9]
        label = LABEL_NAMES.index(data_original[11])
        posision_p = data_original[9:11]
        posision_l = data_original[12:14]
        return i, img[None], label, posision_p, posision_l


class Dataset:
    def __init__(self, mode='train'):
        self.db = VOCBboxDataset(train_datapath)
        self.mode = mode

    def __getitem__(self, idx):
        _, img, label, position_p, posision_l = self.db[idx]

        label = np.array(label)
        if self.mode == 'train':
            return img.copy(), label.copy()
        else:
            return img.copy(), label.copy(), position_p.copy(), posision_l.copy()

    def __len__(self):
        return len(self.db)


def calcorrect(out, label):
    id_list_file = os.path.join(
        train_datapath, '{0}.txt'.format(train_mode))

    ids = [id_.strip() for id_ in open(id_list_file)]
    ecount = dict()
    for i in LABEL_NAMES:
        ecount[i] = 0
    for id in ids:
        ecount[np.loadtxt(train_datapath + str(id) + '.txt')[11]] += 1

    correct = dict()
    error = dict()
    for i in range(0, len(LABEL_NAMES)):
        correct[i] = 0
        error[i] = 0

    pred = torch.argmax(out, dim=1)
    for p, l in zip(pred, y_):
        if (p == l):
            correct[int(l)] += 1
        else:
            error[int(l)] += 1
    precision = sum(correct.values()) / (sum(correct.values()) + sum(error.values()))
    recall = sum(correct.values()) / sum(ecount.values())
    return precision, recall


def calIOU(pre_p, label_p):
    if (pre_p[1] < label_p[0] or pre_p[0] > label_p[1]):
        return 0
    else:
        pred = pre_p.cpu().numpy()
        label = label_p.cpu().numpy()
        res = np.concatenate((pred, label))
        res.sort()
        return (res[2] - res[1]) / (res[3] - res[0])


if train_mode == 'train':
    dataset = Dataset()
    print('load data')
    dataloader = data_.DataLoader(dataset, batch_size=64)

    # 训练模型
    epochs = 200
    model.train()
    for epoch in range(epochs):
        for x_, y_ in dataloader:
            x_ = x_.float().cuda()
            y_ = y_.long().cuda()
            out = model(x_)
            res = calcorrect(out, y_)
            loss = criterion(out, y_)
            print_loss = loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch += 1
            if epoch % 50 == 0:
                print('epoch: {}, loss: {:.4} , 精确率:{} , 召回率:{}'.format(epoch, loss.data.item(), res[0], res[1]))
                torch.save(model.state_dict(), train_datapath + 'parameter.pkl')
            if epoch > 1000000 or print_loss < 0.001:
                print(print_loss)
                print(epoch)
                break

elif train_mode == 'test':
    dataset = Dataset('test')
    print('load data')
    dataloader = data_.DataLoader(dataset, batch_size=64)
    correct = dict()
    error = dict()
    tp = dict()
    fp = dict()
    LA = dict()
    for i in range(0, len(LABEL_NAMES)):
        correct[i] = 0
        error[i] = 0
    for i in range(1, len(LABEL_NAMES)):
        tp[i] = 0
        fp[i] = 0
        LA[i] = 0
    tn = 0
    fn = 0
    # 训练模型
    epoch = 0
    model.eval()

    for x_, y_, position_p, posision_l in dataloader:
        x_ = x_.float()
        y_ = y_.long()
        if torch.cuda.is_available():
            x_ = x_.cuda()
            y_ = y_.cuda()
        else:
            print('GPU不可用')
            # x_ = Variable(x_)
            # y_ = Variable(y_)
        out = model(x_)
        pred = torch.argmax(out, dim=1)
        for p, l in zip(pred, y_):
            if (p == l):
                correct[int(l)] += 1
            else:
                error[int(l)] += 1
        for p, l, pred_p, label_p in zip(pred, y_, position_p, posision_l):
            p = int(p)
            l = int(l)
            if (p == l and l != 0):
                tp[p] += 1
                LA[p] += calIOU(pred_p, label_p)
            elif (p == l and l == 0):
                tn += 1
            elif (p != l and p != 0):
                fp[p] += 1
            elif (p != l and p == 0):
                fn += 1
    precision_sum = sum(tp.values()) / (sum(tp.values()) + sum(fp.values()))
    recall_sum = sum(tp.values()) / (sum(tp.values()) + fn)
    f1_sum = 2 * precision_sum * recall_sum / (precision_sum + recall_sum)
    print('精确率:{} , 召回率:{} , F1:{}'.format(precision_sum, recall_sum, f1_sum))
    precision = dict()
    recall = dict()
    f1 = dict()
    for i in range(1, len(tp) + 1):
        precision[i] = tp[i] / (tp[i] + fp[i])
        recall[i] = tp[i] / (tp[i] + fn)
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        LA[i] /= tp[i]
        print(
            'channel {}    精确率:{} , 召回率:{} , F1:{} ， LA:{}'.format(LABEL_NAMES[i], precision[i], recall[i], f1[i],
                                                                         LA[i]))
    print(correct)
    print(error)

else:  # val
    dataset = Dataset()
    print('load data')
    dataloader = data_.DataLoader(dataset,
                                  batch_size=64,
                                  shuffle=False, )

    correct = dict()
    error = dict()
    for i in range(0, len(LABEL_NAMES)):
        correct[i] = 0
        error[i] = 0

    # 训练模型
    epoch = 0
    model.eval()
    for x_, y_ in dataloader:
        x_ = x_.float()
        y_ = y_.long()
        if torch.cuda.is_available():
            x_ = x_.cuda()
            y_ = y_.cuda()
        else:
            print('GPU不可用')
            # x_ = Variable(x_)
            # y_ = Variable(y_)
        out = model(x_)
        pred = torch.argmax(out, dim=1)
        for p, l in zip(pred, y_):
            if (p == l):
                correct[int(l)] += 1
            else:
                error[int(l)] += 1
    print(correct)
    print(error)
