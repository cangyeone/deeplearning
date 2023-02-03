from ..tensor import Tensor 
import numpy as np 
from .parameter import Parameter 


class Module():
    def __call__(self, *args, **kwds):
        # 正向计算
        return self.forward(*args, **kwds) 
    def parameters(self):
        attr_dict = self.__dict__ 
        pars = [] 
        for key in attr_dict:
            att = attr_dict[key]
            if type(att) == Parameter:
                # 获取模块内可训练参数
                pars.append(att) 
            if hasattr(att, "parameters"):
                # 如果是Module需要递归获取可训练参数
                pars.extend(att.parameters())
        return pars 
    def state_dict(self):
        # 获取变量字典
        attr_dict = self.__dict__ 
        pars = {}
        for key in attr_dict:
            att = attr_dict[key]
            if type(att) == Parameter:
                # 获取模块内可训练参数
                pars[key] = att
            if hasattr(att, "state_dict"):
                # 如果是Module需要递归获取可训练参数
                prepar = att.state_dict() 
                for pkey in prepar:
                    pars[f"{key}.{pkey}"] = prepar[pkey]
        return pars 
    def load_state_dict(self, vdict):
        # 获取变量字典
        pdict = self.state_dict()  
        for key in pdict:
            pdict[key].data = vdict[key].data    
    def load_state_dict_torch(self, vdict):
        # 获取变量字典
        pdict = self.state_dict()  
        for key in pdict:
            pdict[key].data = vdict[key].numpy()

class Sequential(Module):
    def __init__(self, *args) -> None:
        super().__init__()
        self.layers = []
        for i, layer in enumerate(args):
            self.__dict__[f"{i}"] = layer 
            self.layers.append(layer) 
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) 
        return x 

class ModuleList(Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.layers = []
        for i, layer in enumerate(args):
            self.__dict__[f"{i}"] = layer 
            self.layers.append(layer) 
    def __len__(self):
        return len(self.layers) 
    def __getitem__(self, idx):
        return self.layers[idx] 
    