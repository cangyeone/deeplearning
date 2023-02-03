import numpy as np 
from plotconfig import * 

file_ = np.load("data/mnist.npz")
x_train = file_["x_train"]
d_train = file_["y_train"]
x_test = file_["x_test"]
d_test = file_["y_test"]
x_train = x_train.astype(np.float32)/255 
x_test = x_test.astype(np.float32)/255 



def model(x, w, b):
    # 定义模型
    y = x @ w + b 
    return y 

def sigmoid(x):#S函数
    return 1/(1+np.exp(-x))
def dsigmoid(x):#S函数导数
    return np.exp(-x)/(1+np.exp(-x))**2

def grad(x, d, w, b):
    y = model(x, w, b)
    # 加入S函数约束取值范围
    p = sigmoid(y) 
    L = np.sum((p-d)**2, axis=1).mean()
    dLdp = 2 * (p-d) / len(x) 
    dLdy = dLdp * dsigmoid(y)# 参见书中公式 
    dLdw = x.T @ dLdy 
    dLdb = np.sum(dLdy, axis=0) 
    return L, dLdw, dLdb 


x1 = x_train.reshape([-1, 784])
# 独热编码
d1 = np.zeros([len(x1), 10]) 
d1[np.arange(len(x1)), d_train] = 1  

# 定义可训练参数
w = np.random.normal(0, 1, [784, 10])
b = np.zeros([10])
# 超参数定义
eta = 0.1 # 学习率
alpha = 0.01 # 正则化系数
batch_size = 100
n_epoch = 50
for e in range(n_epoch):
    for step in range(len(x_train)//batch_size):
        x = x1[step*batch_size:(step+1)*batch_size] 
        d = d1[step*batch_size:(step+1)*batch_size] 
        loss, gw, gb = grad(x, d, w, b) # 构建模型
        w -= eta * gw + alpha * w 
        b -= eta * gb 

y2 = model(x_test.reshape([-1, 784]), w, b)
p2 = y2.argmax(axis=1) #输出类别
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
          