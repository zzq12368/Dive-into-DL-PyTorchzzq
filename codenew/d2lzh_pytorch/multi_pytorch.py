import torch
import torchvision
import numpy as np
from torch.nn import init
import torch.nn as nn
import sys

sys.path.append("..") # 为了导入上层目录的d2lzh_pytorch
import utils as d2l

num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs),
        )

for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)