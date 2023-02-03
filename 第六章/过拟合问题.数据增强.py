from cv2 import rotate
import numpy as np
from regex import E 
from plotconfig import * 
import torchvision.transforms as transforms

file_ = np.load("data/mnist.npz")
x_train = file_["x_train"]
d_train = file_["y_train"]
x_test = file_["x_test"]
d_test = file_["y_test"]
x_train = x_train
x_test = x_test
print(x_train.max())



import torch 
import torch.nn as nn 
import torch.nn.functional as F 
# 将数据转换为Tensor 
x1 = torch.tensor(x_train[:1000], dtype=torch.float32) / 128-1
d1 = torch.tensor(d_train[:1000], dtype=torch.long) # 整形数据
x2 = torch.tensor(x_test[:1000], dtype=torch.float32) /128-1
d2 = torch.tensor(d_test[:1000], dtype=torch.long) # 整形数据
import torch.nn as nn 
class Model1(nn.Module):
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
            #nn.BatchNorm2d(128), 
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

class Model2(nn.Module):
    def __init__(self):
        super().__init__() 
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), # 16个滤波器，输入1个通道灰度图
            nn.BatchNorm2d(16), 
            nn.ReLU(), # 也需要激活函数获得非线性特征
            nn.Conv2d(16, 16, 3, 1, 1), # 16个滤波器，输入1个通道灰度图
            nn.BatchNorm2d(16), 
            nn.ReLU(), # 也需要激活函数获得非线性特征
            nn.Conv2d(16, 32, 3, 1, 1), # 16个滤波器，输入1个通道灰度图
            nn.BatchNorm2d(32), 
            nn.ReLU(), # 也需要激活函数获得非线性特征
            nn.Conv2d(32, 32, 3, 2, 1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.Conv2d(32, 64, 3, 1, 1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.Conv2d(64, 64, 3, 1, 1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.Conv2d(64, 128, 3, 1, 1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(), 
            nn.Conv2d(128, 128, 3, 1, 1), 
            nn.BatchNorm2d(128), 
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
model1 = Model2()
model2 = Model2()
fig = plt.figure(1, figsize=(12, 9), dpi=100)
gs = grid.GridSpec(1, 2)
names = ["未进行增强", "数据增强后"]
alphas = [0.0, 0.1]
# 数据增强函数
trans = nn.Sequential(
    transforms.RandomRotation([-30, 30]), # 随机旋转
    transforms.RandomResizedCrop([28, 28], scale=(0.8, 1)), # 随机剪裁
)
for itrmodel, model in enumerate([model1, model2]):
    model.train()
    lossfn = nn.CrossEntropyLoss()
    # 定义学习率
    eta = 1e-2
    batch_size = 32
    n_epoch = 10
    alpha = alphas[itrmodel] 
    # 定义优化器，传入所有可训练参数、学习率和正则化（weight_decay）
    optim = torch.optim.SGD(model.parameters(), lr=eta, weight_decay=alpha)
    acc1, acc2 = [], []
    for step in range(1000):
        # 获取数据
        st = step % (len(x1)//batch_size-1)
        x = x1[st*batch_size:(st+1)*batch_size] 
        d = d1[st*batch_size:(st+1)*batch_size]
        # 处理数据
        x = x.reshape([-1, 1, 28, 28])
        if itrmodel==1:
            x = trans(x) 
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
            p = y.argmax(dim=1) 
            a1 = (p==d).float().mean() 
            y2 = model(x2[:1000]) 
            p2 = y2.argmax(dim=1) 
            a2 = (p2==d2[:1000]).float().mean()
            acc1.append(a1) 
            acc2.append(a2)
            print(step, loss, acc1[-1], acc2[-1])
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
    ax.set_title(f"{names[itrmodel]}\n训练集={np.max(acc1)*100:.1f}%\n测试集:{np.max(acc2)*100:.1f}%", x=0.5, y=0.5, va="top", ha="left")
plt.savefig("导出图像/过拟合问题.增强.jpg")
plt.savefig("导出图像/过拟合问题.增强.svg")
print(x_train.shape)
          