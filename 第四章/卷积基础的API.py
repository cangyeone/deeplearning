

import torch 
import torch.nn as nn 
import torch.nn.functional as F 


B = 10 
C = 16 
H = 100 
W = 100 
# 模拟数据
x = torch.randn([10, 16, 100, 100])
w = torch.randn([32, 16, 3, 3])
b = torch.zeros([32])
# 卷积计算
h = F.conv2d(x, w, b, stride=1, padding=1) 