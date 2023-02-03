from .base import Optim
import numpy as np 
class SGD(Optim):
    def __init__(self, parameters, lr=0.1, weight_decay=0.0) -> None:
        super().__init__() 
        self.lr = lr 
        self.weight_decay = weight_decay 
        self.parameters = parameters 
    def step(self):
        # 梯度下降法
        for par in self.parameters:
            #print(par.grad.data.shape, np.abs(par.grad.data).sum(), np.abs(par.data).sum())
            grad = par.grad.data + self.weight_decay * par.data 
            par.data -= self.lr * grad 


class Adam(Optim):
    def __init__(self, parameters, 
                 lr=0.001, beta1=0.9, beta2=0.999, 
                 weight_decay=0.0, epsilon=1e-8):
        super().__init__() 
        self.lr = lr 
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.weight_decay = weight_decay 
        self.epsilon = epsilon 
        self.parameters = parameters 
        self.v = [] # 动量 
        self.s = [] # 梯度平方
        for par in self.parameters:
            self.v.append(np.zeros_like(par.data))
            self.s.append(np.zeros_like(par.data))
        self.num_iter = 1
    def step(self):
        # 执行梯度下降法
        self.num_iter += 1
        for idx, par in enumerate(self.parameters):
            # 加入正则化
            grad = par.grad.data + self.weight_decay * par.data 
            # 其他累积参数
            self.v[idx] = \
                self.beta1 * self.v[idx] + (1-self.beta1) * grad
            self.s[idx] = \
                self.beta2 * self.s[idx] + (1-self.beta2) * grad ** 2  
            vhat = self.v[idx] / (1-self.beta1 ** (self.num_iter)) #修正系数
            shat = self.s[idx] / (1-self.beta2 ** (self.num_iter)) #修正系数
            par.data -= self.lr * vhat / (np.sqrt(shat)+self.epsilon) 