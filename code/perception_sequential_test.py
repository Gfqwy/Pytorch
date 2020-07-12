# -*- coding: utf-8 -*-
import torch
from perception_sequential import Perception
model = Perception(100, 1000, 10)  # 构建类的实例
print(model)  # 打印model结构，会显示Sequential中每一层的具体参数配置

data = torch.randn(100)
output = model(data)
print(output)
print(output.shape)
