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
x1 = torch.tensor(x_train[:], dtype=torch.float32) 
d1 = torch.tensor(d_train[:], dtype=torch.long) # 整形数据
x2 = torch.tensor(x_test[:1000], dtype=torch.float32) 
d2 = torch.tensor(d_test[:1000], dtype=torch.long) # 整形数据

class Model(nn.Module):
    def __init__(self):
        super().__init__() 
        # 卷积层用于提取特征
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.BatchNorm2d(16), #BN层加在激活函数前
            nn.ReLU(), 
            nn.Conv2d(16, 32, 3, 2, 1), 
            nn.BatchNorm2d(32), # 批标准化层加在激活函数前
            nn.ReLU(), 
            nn.Flatten(),  # 展平层
        ) 
        # 输出层用于预测类别，其有九个类
        self.output = nn.Linear(7*7*32, 9)
    def forward(self, x):
        x = x.reshape([-1, 1, 28, 28]) # 数据到所需Shape
        h = self.layers(x) # 提取特征 
        y = self.output(h) # 基于特征此进行分类
        return y 


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__() 
        # 卷积层用于提取特征
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.BatchNorm2d(16), #BN层加在激活函数前
            nn.ReLU(), 
            nn.Conv2d(16, 32, 3, 2, 1), 
            nn.BatchNorm2d(32), # 批标准化层加在激活函数前
            nn.ReLU(), 
            nn.Flatten(),  # 展平层
        ) 
        # 输出层用于预测类别，新的数据需要10个模型
        self.output = nn.Linear(7*7*32, 10)
    def forward(self, x):
        x = x.reshape([-1, 1, 28, 28]) # 数据到所需Shape
        h = self.layers(x) # 提取特征 
        y = self.output(h) # 基于特征此进行分类
        return y 

model1 = Model()
model2 = ModelNew()
model3 = ModelNew()
fig = plt.figure(1, figsize=(12, 12), dpi=100)
gs = grid.GridSpec(2, 2)

lossfn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model1.parameters(), lr=1e-3, weight_decay=0.0)
losses = []
accurcy = []
batch_size = 50
x11 = x1[d1!=9] 
d11 = d1[d1!=9]
for step in range(500):
    st = step % (len(x11)//batch_size-1)
    # 获取数据
    x = x11[st*batch_size:(st+1)*batch_size] 
    d = d11[st*batch_size:(st+1)*batch_size]
    # 处理数据
    x = x.reshape([-1, 784])
    # 标签应当是长整形
    d = d.long()
    # 可以像函数一样调用，需要事先写好forward函数
    y = model1(x) # 构建模型可以直接
    # 计算损失函数
    loss = lossfn(y, d)
    # 计算梯度
    loss.backward()
    optim.step()
    optim.zero_grad()
    if step%10==0:
        print(step, loss)

names = ["重新训练模型", "迁移学习"]
x1 = x1[:500] 
d1 = d1[:500]
for itrmodel, model in enumerate([model2, model3]):
    lossfn = nn.CrossEntropyLoss()
    # 定义学习率
    eta = 0.001
    batch_size = 100
    n_epoch = 10
    alpha = 1e-3 
    
    if itrmodel==1:
        old_ckpt = model1.state_dict()
        new_ckpt = {}
        for k, v in model.named_parameters():
            if "output" in k:
                # 输出层重新训练
                v.requires_grad_(True)
                new_ckpt[k] = v.data 
            else:
                # 卷积层不需要重新训练
                v.requires_grad_(False)
                new_ckpt[k] = old_ckpt[k]
        for k, v in model.named_buffers():
            new_ckpt[k] = old_ckpt[k]
        model.load_state_dict(new_ckpt)
    for k, v in model.named_parameters():
        print(names[itrmodel], k, v.requires_grad, v.std())
    # 定义优化器，传入所有可训练参数、学习率和正则化（weight_decay）
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    losses = []
    accurcy = []
    acc1, acc2 = [], []
    for step in range(50):
        # 获取数据
        st = step % (len(x1)//batch_size-1)
        # 获取数据
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
            #print(step, loss)
            p = y.argmax(dim=1) 
            a1 = (p==d).float().mean() 
            y2 = model(x2[:1000]) 
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
    ax.set_title(f"{names[itrmodel]}\n训练集={np.max(acc1)*100:.1f}%\n测试集:{np.max(acc2)*100:.1f}%", x=0.5, y=0.5, va="top", ha="left")
plt.savefig("导出图像/迁移学习.jpg")
plt.savefig("导出图像/迁移学习.svg")
print(x_train.shape)