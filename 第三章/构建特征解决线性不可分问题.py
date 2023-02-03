
from plotconfig import * 
import numpy as np 

std = 0.3 
r1 = np.random.normal(0, std, [100])
r2 = np.random.normal(1.5, std, [100])
o = np.random.uniform(0, np.pi * 2, [200]) 
r = np.concatenate([r1, r2]) 
x1 = np.stack([r * np.cos(o), r * np.sin(o)]).T 
d1 = np.zeros([200]) 
d1[:100] = 1

x2 = np.random.normal(1, std, [200, 2]) 
x2[50:100] = np.random.normal(-1, std, [50, 2]) 
x2[100:150] = np.random.normal(0, std, [50, 2]) + np.array([-1, 1])
x2[150:200] = np.random.normal(0, std, [50, 2]) + np.array([1, -1])
d2 = np.zeros([200]) 
d2[:100] = 1




import torch 
import torch.nn.functional as F 

def model(x, w, b):
    # 定义模型
    y = x @ w + b 
    return y 
def compute_loss(y, d):
    # 定义损失函数
    e = torch.exp(y) # 计算e指数
    q = e / e.sum(dim=1, keepdim=True) # 归一化
    # 计算损失函数
    loss = -(d*torch.log(q)).sum(dim=1).mean()  
    return loss 

x1 = torch.tensor(x1, dtype=torch.float32)
d1 = torch.tensor(d1, dtype=torch.long) 
d1 = F.one_hot(d1, 2).float() # Onehot标签
# 构建特征 
x1_feature = torch.cat(
    [x1, x1**2, x1[:, 0:1]*x1[:, 1:2]]
    , dim=1) # 矩阵连接，[N, 2+2+1]
# 定义可训练参数
w = torch.randn([5, 2], requires_grad=True) # 权值
b = torch.zeros([2], requires_grad=True) # 偏置，仅影响位置 
# 定义学习率
eta = 0.3 
batch_size = 1
n_epoch = 10
for e in range(n_epoch):
    for step in range(len(x1_feature)//batch_size):
        x = x1_feature[step*batch_size:(step+1)*batch_size] 
        d = d1[step*batch_size:(step+1)*batch_size] 
        y = model(x, w, b) # 构建模型
        loss = compute_loss(y, d)
        # 梯度下降法
        loss.backward() 
        with torch.no_grad():#不需要梯度
            w -= eta * w.grad 
            b -= eta * b.grad 
            w.grad.zero_()  
            b.grad.zero_()

with torch.no_grad():# 推断过程不需要梯度
    y2 = model(x1_feature, w, b)
    p2 = y2.argmax(dim=1) #输出类别
    d2 = d1.argmax(dim=1) 
    print((p2==d2).float().mean())
print(w)