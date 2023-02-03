from ..tensor import Tensor 
import numpy as np 
from .parameter import Parameter 
from ._container import Module
from .functional import relu 
class ReLU(Module):
    def __init__(self) -> None:
        super().__init__() 
    def forward(self, x):
        y = relu(x)
        return y 