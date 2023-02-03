import torch 
import torch.nn as nn 
from utils.wider import ImagetDataset 

class Conv2d(nn.Module):
    def __init__(self, nin=8, nout=11, ks=3, st=1, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(nin, nout, ks, st, padding=padding), 
            nn.BatchNorm2d(nout), 
            nn.ReLU()
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 

class FaceDetection(nn.Module):
    # 全卷积神经网络，不包含全连接层
    def __init__(self):
        super().__init__() 
        self.layers = nn.Sequential(
            Conv2d(3, 32, 3, 2, padding=0), 
            Conv2d(32, 32, 3, 1, padding=0), 
            Conv2d(32, 64, 3, 2, padding=0), 
            Conv2d(64, 64, 3, 1, padding=0), 
            Conv2d(64, 64, 2, 1, 0), 
        )
        self.bbox = nn.Conv2d(64, 4, 1) 
        self.clas = nn.Conv2d(64, 2, 1)
    def forward(self, x):
        x = self.layers(x) 
        box = self.bbox(x) 
        cls = self.clas(x)
        return cls, box 

model = FaceDetection() 
x = torch.zeros([1, 3, 256, 256]) 
c, b = model(x) 
print(c.shape, b.shape)