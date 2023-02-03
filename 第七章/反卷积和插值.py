import torch 
import torch.nn as nn 


K = 3 # 卷积核心大小
S = 2 # 步长
# 二维上采样方式
## 转置卷积
net1 = nn.ConvTranspose2d(64, 32, 
        kernel_size=K, stride=S, 
        padding=(K-1)//2, output_padding=S-1) 
## 上采样：临近插值
net2 = nn.Sequential(
    nn.UpsamplingNearest2d(scale_factor=S), #上采样2倍
    nn.Conv2d(64, 32, K, stride=1, padding=(K-1)//2)
)
## 上采样：线性插值
net3 = nn.Sequential(
    nn.UpsamplingBilinear2d(scale_factor=S), #上采样2倍
    nn.Conv2d(64, 32, K, stride=1, padding=(K-1)//2)
)
x = torch.zeros([10, 64, 10, 10])# 伪造输入 
y1 = net1(x) 
y2 = net2(x) 
y3 = net3(x) 
print(y1.shape, y2.shape, y3.shape)

# 一维上采样方式
## 转置卷积
net1 = nn.ConvTranspose1d(64, 32, 
        kernel_size=K, stride=S, 
        padding=(K-1)//2, output_padding=S-1) 
## 上采样：临近插值
net2 = nn.Sequential(
    nn.Upsample(scale_factor=S, mode="nearest"), #上采样2倍
    nn.Conv1d(64, 32, K, stride=1, padding=(K-1)//2)
)
## 上采样：二次线性插值只能用于二维图像数据
x = torch.zeros([10, 64, 10])# 伪造1D输入 
y1 = net1(x) 
y2 = net2(x) 
print(y1.shape, y2.shape)


# PixelShuffle仅有二维形式
net1 = nn.Sequential(
    nn.Conv2d(64, 32*S**2, K, stride=1, padding=(K-1)//2), 
    nn.PixelShuffle(upscale_factor=S)
)
x = torch.zeros([10, 64, 10, 10])# 伪造输入 
y1 = net1(x) 

# 一维可以通过矩阵变换形式来完成PixelShuffle目标
class PixcelShuffle1D(nn.Module):
    def __init__(self, nin, nout, kernelsize, stride):
        super().__init__() 
        K = kernelsize
        # 定义卷积层
        self.layer = nn.Conv1d(
            nin, nout*S, K, stride=1, padding=(K-1)//2)
        self.S = stride 
        self.C2 = nout 
    def forward(self, x):
        B, C, T = x.shape 
        y = self.layer(x) #[B, C2*S, T] 
        y = y.reshape([B, self.C2, self.S, T])
        y = y.permute(0, 1, 3, 2) # [B, C2, T, S] 
        y = y.reshape([B, self.C2, T*S])  
        return y 
net1 = PixcelShuffle1D(64, 32, K, stride=S)
x = torch.zeros([10, 64, 10])# 伪造输入 
y1 = net1(x) 
print(y1.shape)