import torch 
import torch.nn as nn 

class Conv3(nn.Module):
    # 固定为3的卷积+ReLU
    def __init__(self, nin, nout):
        super().__init__() 
        self.layers = nn.Sequential(
            nn.Conv2d(nin, nout, 3, padding=1), 
            nn.ReLU()
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 

class Block(nn.Module):
    # 多个卷积+最大池化
    def __init__(self, nin, nout, nrepeat):
        super().__init__() 
        self.layers = nn.ModuleList( # 包含多个层
            [Conv3(nin, nout)] + \
            [Conv3(nout, nout) for i in range(nrepeat-1)] + \
            [nn.MaxPool2d(2, 2)] # 降采样
        )
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) 
        return x 

class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            Block(3, 64, 2), Block(64, 128, 2), 
            Block(128, 256, 3), Block(256, 512, 3), 
            Block(512, 512, 3), nn.Flatten(), 
            nn.Linear(7*7*512, 4096), 
            nn.Linear(4096, 4096), 
            nn.Linear(4096, 1000)
        )
    def forward(self, x):
        y = self.layers(x) 
        return y 

model = VGGNet() 
x = torch.ones([1, 3, 224, 224]) 
y = model(x) 
print(y.shape)