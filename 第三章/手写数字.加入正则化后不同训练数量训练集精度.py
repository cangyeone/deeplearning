import numpy as np 
from plotconfig import * 

file_ = np.load("data/mnist.npz")
x_train = file_["x_train"]
d_train = file_["y_train"]
x_test = file_["x_test"]
d_test = file_["y_test"]
x_train = x_train#.astype(np.float32)/255 
x_test = x_test#.astype(np.float32)/255 



import torch 
import torch.nn.functional as F 
# 将数据转换为Tensor 
x1 = torch.tensor(x_train, dtype=torch.float32) 
d1 = torch.tensor(d_train, dtype=torch.long) # 整形数据

x1 = x1.reshape([-1, 784]) 
d1 = d1.reshape([-1])
# 数据预处理
x1 /= 255 # 8bit数据数据最大值为255
d1 = F.one_hot(d1, 10).float() # Onehot编码

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

# 定义可训练参数
w = torch.randn([784, 10], requires_grad=True) # 权值
b = torch.zeros([10], requires_grad=True) # 偏置，仅影响位置 
# 定义学习率
eta = 0.3 
batch_size = 50
n_epoch = 10
weight = []
number = [500, 50000]
acc = []
for num in number:
    www = []
    x2 = x1[:num] 
    d2 = d1[:num]
    for i in range(2):
        w = torch.randn([784, 10], requires_grad=True) # 权值
        b = torch.zeros([10], requires_grad=True) # 偏置，仅影响位置 
        for step in range(5000):
            idx = torch.randint(0, num, [batch_size])
            x = x2[idx] 
            d = d2[idx] 
            y = model(x, w, b) # 构建模型
            loss = compute_loss(y, d) + 0.01 * w.square().sum()
            # 梯度下降法
            loss.backward() 
            with torch.no_grad():#不需要梯度
                w -= eta * w.grad 
                b -= eta * b.grad 
                w.grad.zero_()  
                b.grad.zero_()
        www.append(w.reshape([-1]).detach().numpy())
        with torch.no_grad():# 推断过程不需要梯度
            y2 = model(x2, w, b)
            p1 = y2.argmax(dim=1) #输出类别
            p1 = p1.numpy() # no_grad修饰后不需要detach
        print(f"{num}训练准确度{np.mean(p1==d_train[:num])}")
        x3 = torch.tensor(x_test, dtype=torch.float32) 
        x3 = x3.reshape([-1, 784]) 
        x3 /= 255 # 8bit数据数据最大值为255
        with torch.no_grad():# 推断过程不需要梯度
            y3 = model(x3, w, b)
            p2 = y3.argmax(dim=1) #输出类别
            p2 = p2.numpy() # no_grad修饰后不需要detach
        print(f"{num}预测准确度{np.mean(p2==d_test)}")
    weight.append(www[1]-www[0])
    acc.append(np.mean(p2==d_test))

fig = plt.figure(1, figsize=(12, 9), dpi=100)
gs = grid.GridSpec(1, 1)
color = ["#ff0000", "#0000ff"]
ax = fig.add_subplot(gs[0])
bar, wei = np.histogram(weight[0], bins=33, range=(-5, 5))
width = wei[1]-wei[0]
ax.bar(wei[:-1]+width/2, bar, width=width, color=color[0], alpha=0.5, label=f"{number[0]}个样本训练,精度{acc[0]:.2f}")
bar, wei = np.histogram(weight[1], bins=33, range=(-5, 5))
width = wei[1]-wei[0]
ax.bar(wei[:-1]+width/2, bar, width=width, color=color[1], alpha=0.5, label=f"{number[1]}个样本训练,精度{acc[1]:.2f}")
ax.legend(loc="upper right")
ax.set_title("两次迭代的差$w_1-w_2$")
ax.set_xlim((-5, 5))
plt.savefig("导出图像/手写数字.加正则化.jpg")
plt.savefig("导出图像/手写数字.加正则化.pdf")
print(x_train.shape)
          