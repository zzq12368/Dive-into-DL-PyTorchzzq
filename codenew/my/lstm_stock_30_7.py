import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

df = pd.read_csv('e:/sh.csv', index_col=0)
df.index = list(map(lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'), df.index))


def getData(df, column, train_end=-300, days_before=30, days_pred=7, return_all=True, generate_index=False):
    series = df[column].copy()

    # 创建训练集
    data = pd.DataFrame()
    # 准备天数
    for i in range(days_before):
        # 最后的 -days_before - days_pred 天只是用于预测值，预留出来
        data['b%d' % i] = series.tolist()[i: -days_before - days_pred + i]

    # 预测天数
    for i in range(days_pred):
        data['y%d' % i] = series.tolist()[days_before + i: - days_pred + i]

    # 是否生成 index
    if generate_index:
        data.index = series.index[days_before:]
    train_data, val_data, test_data = data[:train_end - 300], data[train_end - 300:train_end], data[train_end:]

    if return_all:
        return train_data, val_data, test_data, series, df.index.tolist()

    return train_data, val_data, test_data


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
class TrainSet(Dataset):
    def __init__(self, data):
        self.data, self.label = data[:, :-7].float(), data[:, -7:].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
LR = 0.0001
#EPOCH = 1000
EPOCH = 10
TRAIN_END=-300
DAYS_BEFORE=30
DAYS_PRED=7

# 数据集建立
train_data, val_data, test_data, all_series, df_index = getData(df, 'high', days_before=DAYS_BEFORE, days_pred=DAYS_PRED, train_end=TRAIN_END)

# 获取所有原始数据
all_series_test1 = np.array(all_series.copy().tolist())
# 绘制原始数据的图
plt.figure(figsize=(12,8))
plt.plot(df_index, all_series_test1, label='real-data')

# 归一化，便与训练
train_data_numpy = np.array(train_data)
train_mean = np.mean(train_data_numpy)
train_std  = np.std(train_data_numpy)
train_data_numpy = (train_data_numpy - train_mean) / train_std
train_data_tensor = torch.Tensor(train_data_numpy)

val_data_numpy = np.array(val_data)
val_data_numpy = (val_data_numpy - train_mean) / train_std
val_data_tensor = torch.Tensor(val_data_numpy)

test_data_numpy = np.array(train_data)
test_data_numpy = (test_data_numpy - train_mean) / train_std
test_data_tensor = torch.Tensor(test_data_numpy)

# 创建 dataloader
train_set = TrainSet(train_data_tensor)
train_loader = DataLoader(train_set, batch_size=256, shuffle=True)

val_set = TrainSet(val_data_tensor)
val_loader = DataLoader(val_set, batch_size=256, shuffle=True)



def train_rnn(rnn):

    if torch.cuda.is_available():
        rnn = rnn.cuda()

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.MSELoss()

    best_loss = 1000


    if not os.path.exists('weights'):
        os.mkdir('weights')

    for step in range(EPOCH):
        for tx, ty in train_loader:
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

        with torch.no_grad():
            for tx, ty in val_loader:
                if torch.cuda.is_available():
                    tx = tx.cuda()
                    ty = ty.cuda()

                output = rnn(torch.unsqueeze(tx, dim=2))
                loss = loss_func(torch.squeeze(output), ty)

                print('epoch : %d  ' % step, 'val_loss : %.4f' % loss.cpu().item())

            if loss.cpu().item() < best_loss:
                best_loss = loss.cpu().item()
                torch.save(rnn, 'weights/rnn.pkl'.format(loss.cpu().item()))
                print('new model saved at epoch {} with val_loss {}'.format(step, best_loss))

def get_rnn(ifTrain=False):
    rnn = LSTM()
    if ifTrain:
        train_rnn(rnn)
    else:
        rnn = torch.load('weights/rnn.pkl')
    return rnn

rnn = get_rnn(False)
generate_data_train = []
generate_data_test = []

# 测试数据开始的索引
test_start = len(all_series_test1) + TRAIN_END

# 对所有的数据进行相同的归一化
all_series_test1 = (all_series_test1 - train_mean) / train_std
all_series_test1 = torch.Tensor(all_series_test1)

# len(all_series_test1)  # 3448

for i in range(DAYS_BEFORE, len(all_series_test1) - DAYS_PRED, DAYS_PRED):
    x = all_series_test1[i - DAYS_BEFORE:i]
    # 将 x 填充到 (bs, ts, is) 中的 timesteps
    x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=2)

    if torch.cuda.is_available():
        x = x.cuda()

    y = torch.squeeze(rnn(x))

    if i < test_start:
        generate_data_train.append(torch.squeeze(y.cpu()).detach().numpy() * train_std + train_mean)
    else:
        generate_data_test.append(torch.squeeze(y.cpu()).detach().numpy() * train_std + train_mean)

generate_data_train = np.concatenate(generate_data_train, axis=0)
generate_data_test = np.concatenate(generate_data_test, axis=0)

# print(len(generate_data_train))   # 3122
# print(len(generate_data_test))    # 294

plt.figure(figsize=(12, 8))
plt.plot(df_index[DAYS_BEFORE: len(generate_data_train) + DAYS_BEFORE], generate_data_train, 'b',
         label='generate_train')
plt.plot(df_index[TRAIN_END:len(generate_data_test) + TRAIN_END], generate_data_test, 'k', label='generate_test')
plt.plot(df_index, all_series_test1.clone().numpy() * train_std + train_mean, 'r', label='real_data')
plt.legend()
plt.show()


plt.figure(figsize=(10,16))

plt.subplot(2,1,1)
plt.plot(df_index[100 + DAYS_BEFORE: 130 + DAYS_BEFORE], generate_data_train[100: 130], 'b', label='generate_train')
plt.plot(df_index[100 + DAYS_BEFORE: 130 + DAYS_BEFORE], (all_series_test1.clone().numpy()* train_std + train_mean)[100 + DAYS_BEFORE: 130 + DAYS_BEFORE], 'r', label='real_data')
plt.legend()

plt.subplot(2,1,2)
plt.plot(df_index[TRAIN_END + 50: TRAIN_END + 80], generate_data_test[50:80], 'k', label='generate_test')
plt.plot(df_index[TRAIN_END + 50: TRAIN_END + 80], (all_series_test1.clone().numpy()* train_std + train_mean)[TRAIN_END + 50: TRAIN_END + 80], 'r', label='real_data')
plt.legend()

plt.show()