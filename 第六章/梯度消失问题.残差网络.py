from cv2 import rotate
import numpy as np
from regex import E 
from plotconfig import * 

file_ = np.load("data/mnist.npz")
x_train = file_["x_train"]
d_train = file_["y_train"]
x_test = file_["x_test"]
d_test = file_["y_test"]
x_train = x_train.astype(np.float32)/255 
x_test = x_test.astype(np.float32)/255 



import torch 
import torch.nn as nn 
import torch.nn.functional as F 
# 将数据转换为Tensor 
x1 = torch.tensor(x_train, dtype=torch.float32) 
d1 = torch.tensor(d_train, dtype=torch.long) # 整形数据
import torch.nn as nn 
class Conv2d(nn.Module):
    def __init__(
        self, 
        nin, nout, # 输入图像通道数，输出图像通道数（滤波器数量）
        ks=3, stride=1, # 卷积核心，步长
        padding=0,# 补0，图像上下左右补0的值
        ):
        super().__init__() # 初始化
        # 注册可训练参数
        self.register_parameter("weight", 
            nn.parameter.Parameter(torch.randn([nout, nin, ks, ks])))
        self.register_parameter("bias", 
            nn.parameter.Parameter(torch.randn([nout])))
        self.stride = stride 
        # 一般想使得输入输出相等或者为整数倍的话padding=(ks-1)//2
        self.padding = padding 
        if stride == 1 and nin == nout:
            self.res = True
        else:
            self.res = False 
    def forward(self, x):
        y = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        if self.res:
            #print("残差")
            y = y + x 
        return y 

class ResNet(nn.Module):
    def __init__(self, nin, nout, ks, stride=1, padding=1, activation=nn.ReLU):
        super().__init__()
        self.layers = nn.Sequential(
            Conv2d(nin, nout, ks, stride, padding=padding), 
            activation()
        )
        if stride == 1 and nin == nout:
            self.res = True
        else:
            self.res = False 
    def forward(self, x):
        y = self.layers(x) 
        if self.res:
            y = y+x 
        return y 
class Model1(nn.Module):
    def __init__(self):
        super().__init__() 
        self.layers = nn.Sequential(
            ResNet( 1, 16, 3, 2, 1), # 16个滤波器，输入1个通道灰度图
            ResNet(16, 16, 3, 1, 1), # 16个滤波器，输入1个通道灰度图
            ResNet(16, 16, 3, 1, 1), # 16个滤波器，输入1个通道灰度图
            ResNet(16, 16, 3, 1, 1), # 16个滤波器，输入1个通道灰度图
            ResNet(16, 16, 3, 1, 1), # 16个滤波器，输入1个通道灰度图
            ResNet(16, 32, 3, 2, 1), 
            ResNet(32, 32, 3, 1, 1), 
            ResNet(32, 32, 3, 1, 1), 
            ResNet(32, 32, 3, 1, 1), 
            ResNet(32, 32, 3, 1, 1), 
            nn.Flatten(),  # 做成矩阵个格式
            nn.Linear(7*7*32, 10)
        )
        self.init()
    def init(self):
        for var in self.parameters():
            if len(var.shape)<2:continue 
            nn.init.xavier_uniform_(var, gain=nn.init.calculate_gain('relu'))
    def forward(self, x):
        x = x.reshape([-1, 1, 28, 28]) # 数据到所需Shape
        y = self.layers(x) 
        return y 
class Model2(nn.Module):
    def __init__(self):
        super().__init__() 
        self.layers = nn.Sequential(
            ResNet( 1, 16, 3, 2, 1, nn.Sigmoid), # 16个滤波器，输入1个通道灰度图
            ResNet(16, 16, 3, 1, 1, nn.Sigmoid), # 16个滤波器，输入1个通道灰度图
            ResNet(16, 16, 3, 1, 1, nn.Sigmoid), # 16个滤波器，输入1个通道灰度图
            ResNet(16, 16, 3, 1, 1, nn.Sigmoid), # 16个滤波器，输入1个通道灰度图
            ResNet(16, 16, 3, 1, 1, nn.Sigmoid), # 16个滤波器，输入1个通道灰度图
            ResNet(16, 32, 3, 2, 1, nn.Sigmoid), 
            ResNet(32, 32, 3, 1, 1, nn.Sigmoid), 
            ResNet(32, 32, 3, 1, 1, nn.Sigmoid), 
            ResNet(32, 32, 3, 1, 1, nn.Sigmoid), 
            ResNet(32, 32, 3, 1, 1, nn.Sigmoid), 
            nn.Flatten(),  # 做成矩阵个格式
            nn.Linear(7*7*32, 10)
        )
        self.init()
    def init(self):
        for var in self.parameters():
            if len(var.shape)<2:continue 
            nn.init.xavier_uniform_(var, gain=nn.init.calculate_gain('relu'))
    def forward(self, x):
        x = x.reshape([-1, 1, 28, 28]) # 数据到所需Shape
        y = self.layers(x) 
        return y 
model1 = Model2()
model2 = Model1()
fig = plt.figure(1, figsize=(12, 9), dpi=100)
gs = grid.GridSpec(1, 2)
names = ["a)", "b)"]
for itrmodel, model in enumerate([model1, model2]):
    lossfn = nn.CrossEntropyLoss()
    # 定义学习率
    eta = 0.001
    batch_size = 100
    n_epoch = 10
    alpha = 1e-5 
    N = 20
    # 定义优化器，传入所有可训练参数、学习率和正则化（weight_decay）
    optim = torch.optim.Adam(model.parameters(), lr=eta, weight_decay=alpha)
    for step in range(N):
        # 获取数据
        x = x1[step*batch_size:(step+1)*batch_size] 
        d = d1[step*batch_size:(step+1)*batch_size]
        # 处理数据
        x = x.reshape([-1, 784])
        # 标签应当是长整形
        d = d.long()
        # 可以像函数一样调用，需要事先写好forward函数
        y = model(x) # 构建模型可以直接
        # 计算损失函数
        loss = lossfn(y, d)
        # 计算梯度
        loss.backward()
        optim.step()
        if step >N-3:
            break 
        else:
            print(loss)
            optim.zero_grad()
    ax = fig.add_subplot(gs[itrmodel])
    stats = []
    for var in model.parameters():
        #if "weight" not in key:continue 
        #var = model.state_dict()[key]
        #print(len(var.shape))
        if len(var.shape)<=1:continue 
        g = var.grad.numpy() 
        g = g.reshape(-1).astype(np.float32)
        m1 = np.max(np.abs(g)) 
        m2 = np.median(np.abs(g)) 
        m3 = np.mean(np.abs(g))
        stats.append([m1, m2, m3])
    stats = np.array(stats) 
    #ax.plot(stats[:, 0], c="r", label="最大值")
    ax.plot(stats[:, 1], c="k", marker="o", linestyle="-", label="Medium")
    ax.plot(stats[:, 2], c="k", marker="o", linestyle="--", label="Mean")
    ax.set_xticks([i for i in range(len(stats))])
    ax.set_xticklabels([f"L={i+1}" for i in range(len(stats))], rotation=45)
    ax.set_xlabel("Layer")
    ax.set_yscale('log')
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend()
    ax.set_title(names[itrmodel], x=0, y=1, ha="left", va="bottom")
plt.savefig("导出图像/梯度消失问题测试.resnet.png")
plt.savefig("导出图像/梯度消失问题测试.resnet.svg")
print(x_train.shape)
          