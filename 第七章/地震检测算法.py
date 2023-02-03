import torch.nn as nn 
import torch 
import torch.functional as F 
class ConvBNReLU(nn.Module):# 标准的卷积层：Conv+BN+ReLU
    def __init__(self, nin, nout, ks, stride=1):
        super().__init__() 
        pad = (ks-1)//2
        self.layers = nn.Sequential(
            nn.Conv1d(nin, nout, ks, stride, padding=pad), # 一维卷积
            nn.BatchNorm1d(nout), # 一维批标准化层
            nn.ReLU()
        )
    def forward(self, x):
        y = self.layers(x) 
        return y 
class Conv1dTrans(nn.Module):# 上采样+卷积
    def __init__(self, nin, nout, ks, stride):
        super().__init__() 
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=stride, mode="nearest"), 
            ConvBNReLU(nin, nout, ks, stride=1)
        )
    def forward(self, x):
        y = self.layers(x) 
        return y 

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_stride = 8  # 总步长
        F = 16 
        self.layers = nn.Sequential(
            ConvBNReLU(     3, F*2**0, 3, 2),
            ConvBNReLU(F*2**0, F*2**1, 3, 2),
            ConvBNReLU(F*2**1, F*2**1, 3, 1),
            ConvBNReLU(F*2**1, F*2**2, 3, 2),
            ConvBNReLU(F*2**2, F*2**2, 3, 1),
            ConvBNReLU(F*2**2, F*2**3, 3, 2),
            ConvBNReLU(F*2**3, F*2**3, 3, 1)
        )
        self.class_encoder = nn.Sequential(
            ConvBNReLU( F*2**3, F*2**3, 3, 2),
            ConvBNReLU( F*2**3, F*2**3, 3, 2),
            ConvBNReLU( F*2**3, F*2**3, 3, 2),
            Conv1dTrans(F*2**3, F*2**3, 3, 2),
            Conv1dTrans(F*2**3, F*2**3, 3, 2),
            Conv1dTrans(F*2**3, F*2**3, 3, 2),
        )# 本部分相是Yolo中的反馈层，提取高层特征
        self.cl = nn.Conv1d(F * 2 ** 3 * 2, 3, 1)
        self.tm = nn.Conv1d(F * 2 ** 3 * 2, 1, 1)

    def forward(self, x):
        x1 = self.layers(x)
        x2 = self.class_encoder(x1)
        x = torch.cat([x1, x2], dim=1)
        out_class = self.cl(x)
        out_time = self.tm(x)
        out_time = out_time.sigmoid() * self.n_stride
        if self.training:
            return out_class, out_time 
        else: # 如果是推断过程，则输入类别概率，使用softmax 
            out_class = F.softmax(out_class, dim=1) 
            return out_class, out_time 
x = torch.zeros([10, 3, 1024])
model = Model() 
model.train() 
oc, ot = model(x) 
print(oc.shape, ot.shape)

