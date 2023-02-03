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
x2 = torch.tensor(x_test, dtype=torch.float32) 
d2 = torch.tensor(d_test, dtype=torch.long) # 整形数据
import torch.nn as nn 
class Linear(nn.Module):
    def __init__(self, nin, nout):
        super().__init__() # 初始化
        # 注册可训练参数
        self.register_parameter("weight", 
            nn.parameter.Parameter(torch.randn([nin, nout])))
        self.register_parameter("bias", 
            nn.parameter.Parameter(torch.randn([nout])))
    def forward(self, x):
        y = x @ self.weight + self.bias 
        return y 
model = Linear(794, 10)

for var in model.parameters():#在初始化函数中定义的所有可训练参数 
    print(f"Shape:{var.shape}")
#Shape:torch.Size([784, 10])
#Shape:torch.Size([10])
for key in model.state_dict():#在初始化函数中定义的所有可训练参数字典
    var = model.state_dict()[key]
    print(f"名称:{key},Shape:{var.shape}")
#名称:weight,Shape:torch.Size([784, 10])
#名称:bias,Shape:torch.Size([10])
lossfn = nn.CrossEntropyLoss()
# 定义学习率
eta = 0.1 
batch_size = 50
n_epoch = 10
alpha = 1e-3 
# 定义优化器，传入所有可训练参数、学习率和正则化（weight_decay）
optim = torch.optim.SGD(model.parameters(), lr=eta, weight_decay=alpha)
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
        y = model(torch.cat([x, F.one_hot(d, 10)], dim=1)) # 构建模型可以直接
        # 计算损失函数
        loss = lossfn(y, d)
        # 计算梯度
        loss.backward()
        # 执行w-=eta * dw  
        optim.step() 
        # 将所有的可训练参数置零
        optim.zero_grad()
    x2 = x2.reshape([-1, 784])
    y2 = model(torch.cat([x2, F.one_hot(d2, 10)], dim=1))
    p2 = y2.argmax(dim=1) 
    acc = (p2==d2).float().mean() 
    print(acc)