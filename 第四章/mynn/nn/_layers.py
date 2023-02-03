from ..tensor import Tensor 
import numpy as np 
from .parameter import Parameter 
from ._container import Module
from .functional import conv2d 
class Linear(Module):
    def __init__(self, nin, nout) -> None:
        super().__init__() 
        self.weight = Parameter(np.random.normal(0, 1, [nin, nout]), training=True) 
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