import numpy as np 
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

class Model2(nn.Module):
    def __init__(self):
        super().__init__() 
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.BatchNorm2d(16), #BN层加在激活函数前
            nn.Sigmoid(), 
            nn.Conv2d(16, 32, 3, 2, 1), 
            nn.BatchNorm2d(32), # 批标准化层加在激活函数前
            nn.Sigmoid(), 
            nn.Flatten(),  # 展平层
            nn.Linear(7*7*32, 10)
        )
    def forward(self, x):
        x = x.reshape([-1, 1, 28, 28]) # 数据到所需Shape
        y = self.layers(x) 
        return y 

class ConvBNReLU(nn.Module):
    """卷积层中加入批标准化"""
    def __init__(self, nin, nout, ks=3, stride=1):
        super().__init__() 
        pad = (ks-1)//2
        self.conv = nn.Conv2d(nin, nout, stride=stride, padding=pad)
        self.norm = nn.BatchNorm2d(nout) 
        self.relu = nn.ReLU() 
    def forward(self, x):
        x = self.conv(x) 
        x = self.norm(x) 
        x = self.relu(x) 
        return x 

class Model1(nn.Module):
    def __init__(self):
        super().__init__() 
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            #nn.BatchNorm2d(16), #BN层加在激活函数前
            nn.Sigmoid(), 
            nn.Conv2d(16, 32, 3, 2, 1), 
            #nn.BatchNorm2d(32) # 批标准化层加在激活函数前
            nn.Sigmoid(), 
            nn.Flatten(),  # 展平层
            nn.Linear(7*7*32, 10)
        )
    def forward(self, x):
        x = x.reshape([-1, 1, 28, 28]) # 数据到所需Shape
        y = self.layers(x) 
        return y 


class Model3(nn.Module):
    def __init__(self):
        super().__init__() 
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            #nn.BatchNorm2d(16), #BN层加在激活函数前
            nn.Sigmoid(), 
            nn.Conv2d(16, 32, 3, 2, 1), 
            #nn.BatchNorm2d(32) # 批标准化层加在激活函数前
            nn.Sigmoid(), 
            nn.Flatten(),  # 展平层
            nn.Linear(7*7*32, 10)
        )
        self.init()
    def init(self):
        for var in self.parameters():
            if len(var.shape)<2:
                nn.init.zeros_(var) #偏置初始化为0
            else:#其他使用随机初始化,gain即为文章中的a
                nn.init.xavier_uniform_(var, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = x.reshape([-1, 1, 28, 28]) # 数据到所需Shape
        y = self.layers(x) 
        return y 

model1 = Model1()
model2 = Model2()
model3 = Model3()
fig = plt.figure(1, figsize=(12, 9), dpi=100)
gs = grid.GridSpec(2, 3)
names = ["a)", "b)", "c)"]
for itrmodel, model in enumerate([model1, model3, model2]):
    lossfn = nn.CrossEntropyLoss()
    # 定义学习率
    eta = 0.001
    batch_size = 100
    n_epoch = 10
    alpha = 1e-3 
    # 定义优化器，传入所有可训练参数、学习率和正则化（weight_decay）
    optim = torch.optim.Adam(model.parameters(), lr=eta, weight_decay=alpha)
    losses = []
    accurcy = []
    for step in range(500):
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
        optim.zero_grad()
        losses.append(loss.detach().cpu().numpy())
        p = y.argmax(dim=1) 
        acc = (p==d).float().mean().cpu().numpy() 
        accurcy.append(acc)

    ax = fig.add_subplot(gs[0, itrmodel])
    ax.plot(losses, c="k", label="中位数")
    ax.set_xlabel("迭代次数")
    if itrmodel == 0:
        ax.set_ylabel("损失函数")
    ax.set_xlim((0, 500))
    ax.set_ylim((0, 2.5))
    ax.grid(True)
    #ax.legend()
    ax.set_title(names[itrmodel], x=0, y=1, ha="left", va="bottom")
    ax = fig.add_subplot(gs[1, itrmodel])
    ax.plot(accurcy, c="k", label="中位数")
    ax.set_xlabel("迭代次数")
    if itrmodel == 0:
        ax.set_ylabel("准确度")
    ax.set_xlim((0, 500))
    ax.set_ylim((-0.1, 1.1))
    ax.grid(True)
    #ax.legend()
    #ax.set_title(names[itrmodel])
plt.savefig("导出图像/初始化测试.bn.loss.jpg")
plt.savefig("导出图像/初始化测试.bn.loss.svg")
print(x_train.shape)
          