from plotconfig import * 
import numpy as np 

import torch.nn as nn
import torch 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader 
import math 

class SelfAttention(nn.Module):
    """
    自注意力机制
    """
    def __init__(self, nin, units=32):
        super().__init__()
        self.K = nn.Linear(nin, units)
        self.Q = nn.Linear(nin, units)
        self.V = nn.Linear(nin, units)
        self.dk = units 
    def forward(self, x):
        # x:[T, B, C]
        x = x.permute(1, 0, 2)  #T,B,C->B,T,C
        k = self.K(x) # B, T1, C 
        q = self.Q(x).permute(0, 2, 1) # B, C, T2 
        v = self.V(x) # B, T, C

        s = k @ q  # B,T1,T2 
        s /= math.sqrt(self.dk) 
        e = s.softmax(dim=-1) # 需要softmax进行归一化
        y = e @ v   
        y = y.permute(1, 0, 2) #B,T,C->T,B,C
        return y 


k = torch.zeros([10, 20, 32]) 
q = torch.zeros([10, 32, 30]) 
e = k @ q 
print(e.shape)  

x = torch.zeros([20, 10, 32]) 
m = SelfAttention(32, 32) 
y = m(x) 
print(y.shape)
