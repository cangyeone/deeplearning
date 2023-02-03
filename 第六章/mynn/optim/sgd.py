from .base import Optim

class SGD(Optim):
    def __init__(self, parameters, lr=0.1, weight_decay=0.0) -> None:
        super().__init__() 
        self.lr = lr 
        self.weight_decay = weight_decay 
        self.parameters = parameters 
    def step(self):
        # 梯度下降法
        for par in self.parameters:
            grad = par.grad.data + self.weight_decay * par.data 
            par.data -= self.lr * grad 

