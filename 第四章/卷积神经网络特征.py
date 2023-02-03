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
class Model(nn.Module):
    def __init__(self):
        super().__init__() 
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), # 16个滤波器，输入1个通道灰度图
            nn.ReLU(), # 也需要激活函数获得非线性特征
        ) 
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1), 
            nn.ReLU(), 
        ) 
        self.layer3 = nn.Sequential(
            nn.Flatten(),  # 做成矩阵个格式
            nn.Linear(7*7*32, 10)
        )
    def forward(self, x):
        x = x.reshape([-1, 1, 28, 28])
        h1 = self.layer1(x) 
        h2 = self.layer2(h1)
        y = self.layer3(h2) 
        return y, h1, h2  

model = Model()
model.load_state_dict(torch.load("ckpt/mnist.feature.pt"))
lossfn = nn.CrossEntropyLoss()
# 定义学习率
eta = 0.001
batch_size = 50
n_epoch = 1
alpha = 1e-3 
# 定义优化器，传入所有可训练参数、学习率和正则化（weight_decay）
optim = torch.optim.Adam(model.parameters(), lr=eta, weight_decay=alpha)
for e in range(n_epoch):
    for step in range(len(x_train)//batch_size):
        # 获取数据
        x = x1[step*batch_size:(step+1)*batch_size] 
        d = d1[step*batch_size:(step+1)*batch_size]
        # 处理数据
        x = x.reshape([-1, 784])
        # 标签应当是长整形
        d = d.long()
        # 可以像函数一样调用，需要事先写好forward函数
        y, h1, h2 = model(x) # 构建模型可以直接
        # 计算损失函数
        loss = lossfn(y, d)
        # 计算梯度
        loss.backward()
        # 执行w-=eta * dw  
        optim.step() 
        # 将所有的可训练参数置零
        optim.zero_grad()
    # 保存模型
    var_dict = model.state_dict() 
    torch.save(var_dict, "ckpt/mnist.feature.pt")

    print(loss)
    x2 = torch.tensor(x_test, dtype=torch.float32) 
    x2 = x2.reshape([-1, 784]) 
    with torch.no_grad():# 推断过程不需要梯度
        y2, h1, h2 = model(x2)
        p2 = y2.argmax(dim=1) #输出类别
        p2 = p2.numpy() # no_grad修饰后不需要detach
    print(f"预测准确度{np.mean(p2==d_test)}")


x2 = x2.reshape([-1, 28, 28]).cpu().numpy() 
h1 = h1.cpu().numpy() 
h2 = h2.cpu().numpy() 
fig = plt.figure(1, figsize=(12, 12), dpi=100)
gs = grid.GridSpec(2, 2)
gs1 = grid.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs[0, 1])
gs2 = grid.GridSpecFromSubplotSpec(4, 8, subplot_spec=gs[1, :])
ax = fig.add_subplot(gs[0, 0])
midx = 0
ax.matshow(x2[midx], cmap=plt.get_cmap("Greys"))
ax.set_xticks(())
ax.set_yticks(())
ax.set_title("a)", x=0, y=1, va="bottom", ha="left")
ax = fig.add_subplot(gs[0, 1])
ax.set_title("b)", x=0, y=1, va="bottom", ha="left")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks(())
ax.set_yticks(())
ax = fig.add_subplot(gs[1, :])
ax.set_title("c)", x=0, y=1, va="bottom", ha="left")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks(())
ax.set_yticks(())
for i in range(4):
    for j in range(4):
        ax = fig.add_subplot(gs1[i, j])
        ax.matshow(h1[midx, i*4+j], cmap=plt.get_cmap("Greys"))
        ax.set_xticks(())
        ax.set_yticks(())
        ax = fig.add_subplot(gs2[i, j])
        ax.matshow(h2[midx, i*4+j], cmap=plt.get_cmap("Greys"))
        ax.set_xticks(())
        ax.set_yticks(())
        ax = fig.add_subplot(gs2[i, j+4])
        ax.matshow(h2[midx, i*4+j+4], cmap=plt.get_cmap("Greys"))
        ax.set_xticks(())
        ax.set_yticks(())

plt.savefig("导出图像/手写数字.feature.svg")
plt.savefig("导出图像/手写数字.feature.png")
print(x_train.shape)
          