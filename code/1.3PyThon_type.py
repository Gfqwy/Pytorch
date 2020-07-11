# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, ndim):
        super(LinearModel, self).__init__()
        self.ndim = ndim
        self.weight = nn.Parameter(torch.randn(ndim, 1))  # 定义权重
        self.bias = nn.Parameter(torch.randn(1))  # 定义偏置

    def forward(self, x):
        # 定义线性模型 y = Wx + b
        return x.mm(self.weight) + self.bias
