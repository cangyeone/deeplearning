import numpy as np 
import time 
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
# 定义学习率
eta = 0.1
n_epoch = 10
batch_sizes = [1, 5, 10, 30, 60, 120]
batch_size = 1
losses = []
times = []
for batch_size in batch_sizes:
    ls = []
    acc_time = []
    w = torch.randn([784, 10], requires_grad=True) # 权值
    b = torch.zeros([10], requires_grad=True) # 偏置，仅影响位置 
    for step in range(500):
        x = x1[step*batch_size:(step+1)*batch_size] 
        d = d1[step*batch_size:(step+1)*batch_size] 
        t1 = time.perf_counter()
        y = model(x, w, b) # 构建模型
        loss = compute_loss(y, d)
        # 梯度下降法
        loss.backward() 
        with torch.no_grad():#不需要梯度
            w -= eta * w.grad 
            b -= eta * b.grad 
            w.grad.zero_()  
            b.grad.zero_()
        t2 = time.perf_counter()
        if step % 5 ==0:continue 
        x2 = torch.tensor(x_test, dtype=torch.float32) 
        x2 = x2.reshape([-1, 784]) 
        x2 /= 255 # 8bit数据数据最大值为255
        with torch.no_grad():# 推断过程不需要梯度
            y2 = model(x2, w, b)
            p2 = y2.argmax(dim=1) #输出类别
            p2 = p2.numpy() # no_grad修饰后不需要detach   
        ls.append(np.mean(p2==d_test))
        times.append([t2-t1])
    losses.append(ls)    
    print(np.mean(times)*1000, loss)
x2 = torch.tensor(x_test, dtype=torch.float32) 
x2 = x2.reshape([-1, 784]) 
x2 /= 255 # 8bit数据数据最大值为255
with torch.no_grad():# 推断过程不需要梯度
    y2 = model(x2, w, b)
    p2 = y2.argmax(dim=1) #输出类别
    p2 = p2.numpy() # no_grad修饰后不需要detach
print(f"预测准确度{np.mean(p2==d_test)}")
fig = plt.figure(1, figsize=(12, 9), dpi=100)
gs = grid.GridSpec(1, 1)

ax = fig.add_subplot(gs[0])
color = ["#333333", "#666666", "#999999", "#ff0000", "#00ff00", "#0000ff"]
for i in range(6):
    ls = np.array(losses[i])
    bs = batch_sizes[i]
    tm = times[i]
    t = np.arange(len(ls)) * 5 
    ax.plot([], [], c="k", marker=f"${i+1}$", linestyle="--", label=f"批尺寸:{bs}")
    ax.plot(t, ls, c="k", alpha=0.3, linestyle="--")
    ax.scatter(t[::5], ls[::5], c="k", s=50, marker=f"${i+1}$")
ax.legend(loc="lower right")
ax.set_xlabel("迭代次数")
ax.set_ylabel("训练集准确度")
ax.set_ylim((0, 1))
plt.savefig("导出图像/手写数字.不同批尺寸.svg")
plt.savefig("导出图像/手写数字.不同批尺寸.png")
print(x_train.shape)
          