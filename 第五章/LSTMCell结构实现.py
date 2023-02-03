import torch
import torch.nn as nn 

class LSTMCell(nn.Module):
    # 单层LSTM，以单个时间步作为输入
    def __init__(self, nin, nout):
        super().__init__() 
        # 3个门+1个输出=总共四个输出
        self.layer = nn.Linear(nin+nout, 4*nout)
        self.nout = nout 
    def forward(self, xt, state):
        # 某个时间步输入
        ht, ct = state 
        x = torch.cat([xt, ht], dim=1) 
        h = self.layer(x) 
        g1, g2, g3, o = torch.split(h, self.nout, dim=1) 
        g1 = g1.sigmoid() # 遗忘门
        g2 = g2.sigmoid() # 输入门
        g3 = g3.sigmoid() # 输出门 
        o = o.tanh     # 输出
        ct1 = ct * g1 + o * g2 
        ht1 = torch.tanh(ct1) * g3 
        return ht1, (ht1, ct1)

class LSTM(nn.Module):
    # 多层LSTM
    def __init__(self, nin, nout, nlayer):
        super().__init__() 
        self.layers = nn.ModuleList(
            [LSTMCell(nin, nout)] + \
     [LSTMCell(nout, nout) for i in range(nlayer)]
        )
    def forward(self, x, h):
        T, B, C = x.shape 
        ht, ct = h # 获取初始状态向量
        outputs = []
        for t in range(T):
            xt = x[t]
            # 每个时间步依次输入多个层
            for idx, layer in enumerate(self.layers):
                # 输入到idx层网络中
                xt, (ht_layer, ct_layer) = layer(xt, (ht[idx], ct[idx]))
                ht[idx] = ht_layer 
                ct[idx] = ct_layer 
            outputs.append(xt) # 最后一层输出作为神经网络输出
        outputs = torch.stack(outputs, dim=0) # 时间步维度连接
        return outputs, (ht, ct) 
    
        
