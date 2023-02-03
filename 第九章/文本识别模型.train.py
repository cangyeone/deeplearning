from unicodedata import bidirectional
import torch 
import torch.nn as nn 

class ConvBNReLU(nn.Module):
    def __init__(self, nin, nout, stride, ks):
        super().__init__()
        # 图像中为二维卷积
        self.layers = nn.Sequential(
            nn.Conv2d(nin, nout, ks, stride=stride, padding=(ks-1)//2), 
            nn.BatchNorm2d(nout), 
            nn.LeakyReLU(), 
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 

class TextReco(nn.Module):
    def __init__(self, n_words):
        super().__init__() 
        # 卷积神经网络处理图像，降采样后高度为1像素
        self.cnn = nn.Sequential(            # 卷积后图像大小
            ConvBNReLU(  3,   8, [2, 1], 3), # [B,   8, 64, T] 
            ConvBNReLU(  8,  16, [2, 1], 3), # [B,  16, 32, T] 
            ConvBNReLU( 16,  32, [2, 1], 3), # [B,  32, 16, T] 
            ConvBNReLU( 32,  64, [2, 1], 3), # [B,  64,  8, T] 
            ConvBNReLU( 64, 128, [2, 1], 3), # [B, 128,  4, T] 
            ConvBNReLU(128, 256, [2, 1], 3), # [B, 256,  2, T] 
            ConvBNReLU(256, 256, [2, 1], 3), # [B, 256,  1, T] 
        )
        # 循环网络
        self.rnn = nn.GRU(256, 256, 2, bidirectional=True)
        self.out = nn.Linear(512, n_words)
    def forward(self, x):
        B, C, W, T = x.shape 
        x = self.cnn(x) #[B, 256, 1, T] 
        print(x.shape)
        x = x.squeeze().permute(2, 0, 1) 
        h0 = torch.zeros([4, B, 256], device=x.device)
        y, hT = self.rnn(x, h0) 
        y = self.out(y) 
        return y 
model = TextReco(5000)
x = torch.randn([10, 3, 128, 1000])
y = model(x) 
print(y.shape)