# -*- coding: utf-8 -*-
import torch
from torch import nn


# 首先建立一个全连接的子module，继承nn.Module
class Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Linear, self).__init__()  # 调用nn.Module的构造函数
        # 使用nn.Parameter来构造需要学习的参数
        self.w = nn.Parameter(torch.randn(in_dim, out_dim))
        self.b = nn.Parameter(torch.randn(out_dim))

# 在forward中实现前向传播过程
    def forward(self, x):
        x = x.matmul(self.w)  # 使用Tensor.matmul实现矩阵相乘
        y = x + self.b.expand_as(x)  # 使用Tensor.expand_as()来保证矩阵形状一致
        # print('Linear.forward:', y)
        return y


# 构建感知机类，继承nn.Module，并调用了Linear的子module
class Perception(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(Perception, self).__init__()
        self.layer1 = Linear(in_dim, hid_dim)
        self.layer2 = Linear(hid_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)    # 结果输出：Linear里的forward的return y，可通过下一行代码查看
        # print('对比Linear.forward输出的值:', x)
        y = torch.sigmoid(x)  # 使用torch中的sigmoid作为激活函数
        y = self.layer2(y)    # 把使用了激活的x(即y)作为输入放到Linear的forward中得到新的y
        # print('对比最新的Linear.forward输出的值:', y)
        y2 = torch.sigmoid(y)  # 使用torch中的sigmoid作为激活函数
        return y2  # 传入的参数，进行了两次的（wx+b 和 激活函数）
