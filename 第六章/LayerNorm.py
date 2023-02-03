import torch 
import torch.nn as nn 
class LayerNorm(nn.Module):
    def __init__(self, shapes, epsilon=1e-4, affine=True):
        super().__init__() 
        # 仅有可训练参数，数据的均值和方差
        if type(shapes) == int:
            shapes = [shapes]
        if affine:# 如果不进行仿射变换则无可训练参数
            self.register_parameter(
                "weight", nn.Parameter(torch.randn(shapes))
            )
            self.register_parameter(
                "bias", nn.Parameter(torch.randn(shapes))
            )
        self.epsilon = epsilon 
        self.shapes = shapes 
        self.affine = affine 
    def forward(self, x):
        # 训练和推断过程相同
        dim = [-1-i for i in range(len(self.shapes))]
        dim = tuple(dim)
        mean = (x-x.mean(dim=dim, keepdim=True))
        std = (x.var(dim=dim, keepdim=True)+self.epsilon).sqrt() 
        x = (x-mean)/std 
        if self.affine:#如果进行仿射变换需要对数据振幅和均值进行估计
            x = x * self.weight + self.bias 
        return x 

norm = LayerNorm(10)
x = torch.randn([20, 5, 10])
y = norm(x)