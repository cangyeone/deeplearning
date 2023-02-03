
from plotconfig import * 
import numpy as np 
# 制作数据集
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

def model(x, w1, b1, w2, b2):
    # 定义模型
    h = x @ w1 + b1 
    ha = torch.tanh(h) 
    y = ha @ w2 + b2 
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
# 定义可训练参数
w1 = torch.randn([2, 4], requires_grad=True) # 权值
b1 = torch.zeros([4], requires_grad=True) # 偏置，仅影响位置 
w2 = torch.randn([4, 2], requires_grad=True) # 权值
b2 = torch.zeros([2], requires_grad=True) # 偏置，仅影响位置 
# 定义学习率
eta = 0.3 
batch_size = 10 
n_epoch = 1000
for e in range(n_epoch):
    for step in range(len(x1)//batch_size):
        # 不需要构建特征
        x = x1[step*batch_size:(step+1)*batch_size] 
        d = d1[step*batch_size:(step+1)*batch_size] 
        y = model(x, w1, b1, w2, b2) # 构建模型
        loss = compute_loss(y, d)
        # 梯度下降法
        loss.backward() 
        with torch.no_grad():#不需要梯度
            for w in [w1, b1, w2, b2]:
                w -= eta * w.grad 
                w.grad.zero_()  

with torch.no_grad():# 推断过程不需要梯度
    y2 = model(x1, w1, b1, w2, b2)
    p2 = y2.argmax(dim=1) #输出类别
    d2 = d1.argmax(dim=1) 
    # 输出准却度
    print((p2==d2).float().mean())