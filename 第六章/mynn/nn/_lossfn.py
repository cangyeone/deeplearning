from ..tensor import Tensor 
import numpy as np 
from .parameter import Parameter 
from ._container import Module
from .functional import cross_entropy 

class CrossEntropyLoss(Module):
    def __init__(self) -> None:
        super().__init__() 
        pass 
    def forward(self, x, d):
        y = cross_entropy(x, d)
        return y 