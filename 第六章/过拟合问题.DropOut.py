from re import I
from cv2 import rotate
import numpy as np
from regex import E 
from plotconfig import * 

file_ = np.load("data/mnist.npz")
x_train = file_["x_train"]
d_train = file_["y_train"]
x_test = file_["x_test"]
d_test = file_["y_test"]
x_train = x_train.astype(np.float32)/128 - 1
x_test = x_test.astype(np.float32)/128 - 1 
print(x_train.max())



import torch 
import torch.nn as nn 
import torch.nn.functional as F 
# 将数据转换为Tensor 
x1 = torch.tensor(x_train[:1000], dtype=torch.float32) 
d1 = torch.tensor(d_train[:1000], dtype=torch.long) # 整形数据
x2 = torch.tensor(x_test[:1000], dtype=torch.float32) 
d2 = torch.tensor(d_test[:1000], dtype=torch.long) # 整形数据

class Dropout2d(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__() 
        self.dropout = dropout
    def forward(self, x):
        if self.training:
            N, C, W, H = x.shape 
            p = 1-self.dropout 
            r = torch.rand([N, C, 1, 1]) + p 
            r = torch.floor(r).float() 
            out = (x * r)/p 
        else:
            out = x 
        return out  

class Dropout(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__() 
        self.dropout = dropout
    def forward(self, x):
        if self.training:
            N, C = x.shape 
            p = 1-self.dropout 
            r = torch.rand([N, C]) + p #随机取0
            r = torch.floor(r).float() 
            out = (x * r)/p # 这里直接进行处理
        else:
            out = x # 推断过程中直接进行输出
        return out  

import torch.nn as nn 
class Model1(nn.Module):
    def __init__(self, dropout=0.8):
        super().__init__() 
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), # 16个滤波器，输入1个通道灰度图
            nn.Dropout2d(dropout), # DropOut层，训练和测试过程不同
            nn.ReLU(), # 也需要激活函数获得非线性特征
            nn.Conv2d(16, 16, 3, 1, 1), # 16个滤波器，输入1个通道灰度图
            nn.Dropout2d(dropout),
            nn.ReLU(), # 也需要激活函数获得非线性特征
            nn.Conv2d(16, 32, 3, 1, 1), # 16个滤波器，输入1个通道灰度图
            nn.Dropout2d(dropout),
            nn.ReLU(), # 也需要激活函数获得非线性特征
            nn.Conv2d(32, 32, 3, 2, 1), 
            nn.Dropout2d(dropout),
            nn.ReLU(), 
            nn.Conv2d(32, 64, 3, 1, 1), 
            nn.Dropout2d(dropout),
            nn.ReLU(), 
            nn.Conv2d(64, 64, 3, 1, 1), 
            nn.Dropout2d(dropout),
            nn.ReLU(), 
            nn.Conv2d(64, 128, 3, 1, 1), 
            nn.Dropout2d(dropout),
            nn.ReLU(), 
            nn.Conv2d(128, 128, 3, 1, 1), 
            nn.Dropout2d(dropout),
            nn.ReLU(), 
            nn.Flatten(),  # 做成矩阵个格式
            nn.Linear(7*7*128, 10)
        )
        self.init()
    def init(self):
        for var in self.parameters():
            if len(var.shape)<2:
                nn.init.zeros_(var) #偏置初始化为0
            else:#其他使用随机初始化
                nn.init.xavier_uniform_(var, gain=nn.init.calculate_gain('relu'))
    def forward(self, x):
        x = x.reshape([-1, 1, 28, 28]) # 数据到所需Shape
        y = self.layers(x) 
        return y 

class Model2(nn.Module):
    def __init__(self):
        super().__init__() 
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), # 16个滤波器，输入1个通道灰度图
            #nn.BatchNorm2d(16), 
            nn.ReLU(), # 也需要激活函数获得非线性特征
            nn.Conv2d(16, 16, 3, 1, 1), # 16个滤波器，输入1个通道灰度图
            #nn.BatchNorm2d(16), 
            nn.ReLU(), # 也需要激活函数获得非线性特征
            nn.Conv2d(16, 32, 3, 1, 1), # 16个滤波器，输入1个通道灰度图
            #nn.BatchNorm2d(32), 
            nn.ReLU(), # 也需要激活函数获得非线性特征
            nn.Conv2d(32, 32, 3, 2, 1), 
            #nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.Conv2d(32, 64, 3, 1, 1), 
            #nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.Conv2d(64, 64, 3, 1, 1), 
            #nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.Conv2d(64, 128, 3, 1, 1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(), 
            nn.Conv2d(128, 128, 3, 1, 1), 
            #nn.BatchNorm2d(128), 
            nn.ReLU(), 
            nn.Flatten(),  # 做成矩阵个格式
            nn.Linear(7*7*128, 10)
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
model1 = Model1(0.0)
model2 = Model1(0.2)

model3 = Model1(0.0)
model4 = Model1(0.2)
model3.eval() 
model4.eval()
fig = plt.figure(1, figsize=(12, 9), dpi=100)
gs = grid.GridSpec(1, 2)
names = ["", "加入BN"]
alphas = [0.0, 0.0]
infermodel = [model3, model4]
DP = [0, 0.2]
for itrmodel, model in enumerate([model1, model2]):
    model.train()
    
    lossfn = nn.CrossEntropyLoss()
    # 定义学习率
    eta = 1e-2
    batch_size = 50
    n_epoch = 10
    alpha = alphas[itrmodel] 
    # 定义优化器，传入所有可训练参数、学习率和正则化（weight_decay）
    optim = torch.optim.SGD(model.parameters(), lr=eta, weight_decay=alpha)
    acc1, acc2 = [], []
    dropout = DP[itrmodel]
    for step in range(2000):
        # 获取数据

        st = step % (len(x1)//batch_size-1)
        x = x1[st*batch_size:(st+1)*batch_size] 
        d = d1[st*batch_size:(st+1)*batch_size]
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
        optim.zero_grad() 
        if step % 10 == 0:
            print(step, loss)
            infermodel[itrmodel].load_state_dict(model.state_dict())
            with torch.no_grad():
                y = infermodel[itrmodel](x)
                p = y.argmax(dim=1) 
                a1 = (p==d).float().mean() 
                y2 = infermodel[itrmodel](x2[:1000]) 
                p2 = y2.argmax(dim=1) 
                a2 = (p2==d2[:1000]).float().mean()
            acc1.append(a1) 
            acc2.append(a2)
    ax = fig.add_subplot(gs[itrmodel])
    #ax.plot(stats[:, 0], c="r", label="最大值")
    step = np.arange(len(acc1)) * 10 
    ax.plot(step, acc1, c="k", marker="o", linestyle="-", label="训练集")
    ax.plot(step, acc2, c="k", marker="o", linestyle="--", label="测试集")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("准确度")
    ax.set_ylim((-0.0, 1.0))
    ax.grid(True)
    ax.legend(loc="lower right")
    ax.set_title(f"Dropout={dropout:.1f}\n训练集={np.max(acc1)*100:.1f}%\n测试集:{np.max(acc2)*100:.1f}%", x=0.5, y=0.5, va="top", ha="left")
plt.savefig("导出图像/过拟合问题.DropOut.jpg")
plt.savefig("导出图像/过拟合问题.DropOut.svg")
print(x_train.shape)
          