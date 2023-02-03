import numpy as np 
from plotconfig import * 

file_ = np.load("data/mnist.npz")
x_train = file_["x_train"]
d_train = file_["y_train"]
x_test = file_["x_test"]
d_test = file_["y_test"]
x_train = x_train.astype(np.float32)/255 
x_test = x_test.astype(np.float32)/255 


import mynn 
import mynn.nn.functional as F 
x1 = x_train.reshape([-1, 784])
x1 = mynn.Tensor(x1) # 转换为Tensor类型
d1 = d_train.reshape([-1])
d1 = mynn.Tensor(d1) # 转换为Tensor类型

# 定义可训练参数
w1 = np.random.normal(0, 1, [784, 32])
b1 = np.zeros([32])
w1 = mynn.nn.parameter.Parameter(w1, training=True) 
b1 = mynn.nn.parameter.Parameter(b1, training=True)

w2 = np.random.normal(0, 1, [32, 10])
b2 = np.zeros([10])
w2 = mynn.nn.parameter.Parameter(w2, training=True) 
b2 = mynn.nn.parameter.Parameter(b2, training=True)

# 超参数定义
eta = 0.3 # 学习率
alpha = 0.00 # 正则化系数
batch_size = 100
n_epoch = 50
for e in range(n_epoch):
    for step in range(len(x_train)//batch_size):
        x = x1[step*batch_size:(step+1)*batch_size] 
        d = d1[step*batch_size:(step+1)*batch_size] 
        h = F.relu(x @ w1 + b1) 
        y = h @ w2 + b2 
        loss = F.cross_entropy(y, d)
        loss.backward() 
        
        w1.data -= eta * w1.grad.data
        b1.data -= eta * b1.grad.data 
        w2.data -= eta * w2.grad.data
        b2.data -= eta * b2.grad.data 
        
        w1.zero_grad()
        b1.zero_grad()
        w2.zero_grad()
        b2.zero_grad()
    print(loss, np.max(np.abs(w1.data)))

x2 = mynn.Tensor(x_test.reshape([-1, 784]))
h = F.relu(x2 @ w1 + b1) 
y2 = h @ w2 + b2 
p2 = y2.data.argmax(axis=1) #输出类别
print(p2, x_test.reshape([-1, 784]).shape, x2.data.shape, p2.shape)
print(f"预测准确度{np.mean(p2==d_test)}")
fig = plt.figure(1, figsize=(12, 9), dpi=100)
gs = grid.GridSpec(3, 4)
for i in range(3):
    for j in range(4):
        x = x_test[i*4+j]
        d = d_test[i*4+j]
        p = p2[i*4+j]
        ax = fig.add_subplot(gs[i, j])
        ax.matshow(x, cmap=plt.get_cmap("Greys"))
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_xlabel(f"真实类别:{d},预测类别:{p}")
plt.savefig("导出图像/手写数字.torch.jpg")
plt.savefig("导出图像/手写数字.torch.pdf")
print(x_train.shape)
          