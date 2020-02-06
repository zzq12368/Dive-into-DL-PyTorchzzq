import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

LR = 0.0001
#EPOCH = 1000
EPOCH = 10
DAYS_BEFORE=30
DAYS_PRED=7

class DataDeal():
    def __init__(self):

        self.TTdata = pd.DataFrame()
        self.getData()

    def getData(self):
        self.df = pd.read_csv('e:/sh.csv', index_col=0)
        self.allDay = list(map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'), self.df.index))
        self.allData = self.df['high'].copy().tolist()

        # 准备天数
        for i in range(DAYS_BEFORE):
            # 最后的 -days_before - days_pred 天只是用于预测值，预留出来
            self.TTdata['b%d' % i] = self.allData[i: -DAYS_BEFORE - DAYS_PRED + i]

        # 预测天数
        for i in range(DAYS_PRED):
            self.TTdata['y%d' % i] = self.allData[DAYS_BEFORE + i: - DAYS_PRED + i]
        self.tran_data = self.TTdata[: -300]
        self.test_data = self.TTdata[-300:]


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=1,  # 输入尺寸为 1，表示一天的数据
            hidden_size=128,
            num_layers=1,
            batch_first=True)

        self.out = nn.Sequential(
            nn.Linear(128, 1))

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)  # None 表示 hidden state 会用全 0 的 state
        out = self.out(r_out[:, -7:, :])  # 取最后一天作为输出

        return out




class normal_():
    def __init__(self,data):
        self.prevData = np.array(data)
        self.normal_to()
    def normal_to(self):
        self._mean = np.mean(self.prevData)
        self._std = np.std(self.prevData)
        data_normal = (self.prevData - self._mean) / self._std
        self.data_tensor = torch.Tensor(data_normal)


class TrainSet(Dataset):
    def __init__(self, data):
        self.data, self.label = data[:, :-7].float(), data[:, -7:].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
class DataSet_():
    def __init__(self,_tensor):
        train_set = TrainSet(_tensor)
        self.data_iter = train_loader = DataLoader(train_set, batch_size=256, shuffle=True)


def train_rnn(rnn,tran_data_iter):

    if torch.cuda.is_available():
        rnn = rnn.cuda()

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.MSELoss()



    if not os.path.exists('weights'):
        os.mkdir('weights')

    for step in range(EPOCH):
        for tx, ty in tran_data_iter:
            if torch.cuda.is_available():
                tx = tx.cuda()
                ty = ty.cuda()
            inputs = torch.unsqueeze(tx, dim=2)
            output = rnn(inputs)
            loss = loss_func(torch.squeeze(output), ty)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # back propagation, compute gradients
            optimizer.step()
            print('epoch : %d  ' % step, 'train_loss : %.4f' % loss.cpu().item())
    if loss.cpu().item() < 1:
        best_loss = loss.cpu().item()
        torch.save(rnn, 'weights/rnn.pkl'.format(loss.cpu().item()))
        print('new model saved at epoch {} with val_loss {}'.format(step, best_loss))



def comput_rnn(data,train_mean,train_std):
    #标准化
    all_series_test1 = (data - train_mean) / train_std
    all_series_test1 = torch.Tensor(all_series_test1)

    y_return = []
    for i in range(DAYS_BEFORE, len(all_series_test1) - DAYS_PRED, DAYS_PRED):
        x = all_series_test1[i - DAYS_BEFORE:i]
        # 将 x 填充到 (bs, ts, is) 中的 timesteps
        x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=2)

        if torch.cuda.is_available():
            x = x.cuda()

        y = torch.squeeze(rnn(x))
        y_from_normal = torch.squeeze(y.cpu()).detach().numpy() * train_std + train_mean
        y_return.append(y_from_normal)
    y_return = np.concatenate(y_return, axis=0)
    return y_return

ifTrain = False
if __name__ == '__main__':
    rnn = LSTM()
    ds = DataDeal()
    train_normal = normal_(ds.tran_data)
    if ifTrain:

        d_iter = DataSet_(train_normal.data_tensor).data_iter
        train_rnn(rnn, d_iter)
    else:
        rnn = torch.load('weights/rnn.pkl')
    y_out = comput_rnn(ds.allData,train_normal._mean,train_normal._std)

    y_out_len = len(y_out)
    plt.figure(figsize=(12, 8))
    plt.plot(ds.allDay[DAYS_BEFORE: y_out_len-300+DAYS_BEFORE], y_out[0:-300], 'b',label='generate_train')
    plt.plot(ds.allDay[-300:], y_out[-300:], 'k', label='generate_test')
    plt.plot(ds.allDay, ds.allData, 'r', label='real_data')
    plt.legend()
    plt.show()
    print('END')

#-------------------------------------------------------------------------------------------------
