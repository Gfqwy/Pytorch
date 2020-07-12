# -*- coding: utf-8 -*-
# 调用Perception函数
import torch
from perception import Perception
perception = Perception(2, 3, 2)
for name, parameter in perception.named_parameters():
    print(name, parameter)
data = torch.rand(4, 2)
output = perception(data)
print(output)
