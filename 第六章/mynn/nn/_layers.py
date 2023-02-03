from ..tensor import Tensor 
import numpy as np 
from .parameter import Parameter 
from ._container import Module, Sequential, ModuleList 
from .functional import conv2d 
from .functional import relu, tanh 
from ._activation import Tanh 
from ..operator._tensor import cat, split

class Linear(Module):
    def __init__(self, nin, nout) -> None:
        super().__init__() 
        self.weight = Parameter(np.random.normal(0, 1/np.sqrt(nin+nout), [nin, nout]), training=True) 
        self.bias = Parameter(np.zeros([nout]), training=True) 
    def forward(self, x):
        y = x @ self.weight + self.bias 
        return y 

class Conv2d(Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding=0) -> None:
        super().__init__() 
        self.weight = Parameter(
            np.random.normal(0, 0.1, [nout, nin, kernel_size, kernel_size]), training=True) 
        self.bias = Parameter(np.zeros([1, nout, 1, 1]), training=True) 
        self.stride = stride 
        self.pad = padding 
    def forward(self, x):
        y = conv2d(x, self.weight, self.stride, self.pad)
        y = y + self.bias 
        return y 
class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim) -> None:
        super().__init__()
        self.weight = Parameter(
            np.random.normal(0, 0.001, [num_embeddings, embedding_dim]), training=True) 
        self.nword = num_embeddings  
        self.nemb = embedding_dim
    def forward(self, x):
        x = x.data 
        xshape = x.shape 
        outshape = [i for i in xshape] + [self.nemb]
        xl = x.reshape(-1) 
        xonehot = np.zeros([len(xl), self.nword])      
        xonehot[np.arange(len(xl)), xl] = 1 
        xone = Tensor(xonehot)
        xemb = xone @ self.weight 
        xemb = xemb.reshape(outshape) 
        return xemb 


class RNNCell(Module):
    def __init__(self, nin, nout) -> None:
        super().__init__() 
        self.rnn = Sequential(
            Linear(nin+nout, nout), 
            Tanh() # 选择有上下边界（饱和）的激活函数
        )
    def forward(self, x, h):
        xt = cat([x, h], dim=1)
        yt, ht = self.rnn(xt)
        return yt, ht 

class RNN(Module):
    def __init__(self, nin, nout, nlayer) -> None:
        super().__init__() 
        self.layers = []
        for n in range(nlayer):
            self.layers.append(RNNCell(nin, nout))
            nin = nout 
    def forward(self, x, h0):
        T, B, C = x.data.shape 
        outputs = []
        # 每一层的状态向量 
        hts = [h0[i] for i in range(len(self.layers))]
        for t in range(T):
            xt = x[t]
            # 每个时间步依次输入多个层
            newht = []
            for idx, layer in enumerate(self.layers):
                # 输入到idx层网络中
                xt, ht_layer = layer(xt, hts[idx])
                newht.append(ht_layer) 
            hts = newht 
            outputs.append(xt) # 最后一层输出作为神经网络输出
        return outputs, hts     

class LSTMCell(Module):
    def __init__(self, nin, nout) -> None:
        super().__init__() 
        self.rnn = Sequential(
            Linear(nin+nout, nout*4), 
            Tanh() # 选择有上下边界（饱和）的激活函数
        )
        self.nout = nout 
    def forward(self, xt, state):
        # 某个时间步输入
        ht, ct = state 
        #print("数据", xt.shape, ht.shape, ct.shape)
        x = cat([xt, ht], dim=1) 
        h = self.rnn(x) 
        temp = split(h, 4, dim=1) 
        g1, g2, g3, o = temp
        g1 = g1.sigmoid() # 遗忘门
        g2 = g2.sigmoid() # 输入门
        g3 = g3.sigmoid() # 输出门 
        o = o.tanh()     # 输出
        ct1 = ct * g1 + o * g2 
        ht1 = ct1.tanh() * g3 
        return ht1, (ht1, ct1)


class LSTM(Module):
    def __init__(self, nin, nout, nlayer) -> None:
        super().__init__() 
        layers = []
        for n in range(nlayer):
            layers.append(LSTMCell(nin, nout))
            nin = nout 
        self.layers = ModuleList(layers)
    def forward(self, x, h):
        T, B, C = x.data.shape 
        outputs = []
        h0, c0 = h
        # 每一层的状态向量 
        hts = [h0[i] for i in range(len(self.layers))]
        cts = [c0[i] for i in range(len(self.layers))]
        for t in range(T):
            xt = x[t]
            # 每个时间步依次输入多个层
            newht, newct = [], []
            for idx in range(len(self.layers)):
                layer = self.layers[idx]
                # 输入到idx层网络中
                xt, (ht_layer, ct_layer) = layer(xt, (hts[idx], cts[idx]))
                newht.append(ht_layer) 
                newct.append(ct_layer) 
            hts = newht 
            cts = newct 
            outputs.append(xt) # 最后一层输出作为神经网络输出
        return outputs, hts   