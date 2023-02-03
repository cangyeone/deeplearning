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
for e in range(n_epoch):
    for step in range(len(x_train)//batch_size):
        x = x1[step*batch_size:(step+1)*batch_size] 
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

print(loss)
x2 = torch.tensor(x_test, dtype=torch.float32) 
x2 = x2.reshape([-1, 784]) 
x2 /= 255 # 8bit数据数据最大值为255
with torch.no_grad():# 推断过程不需要梯度
    y2 = model(x2, w, b)
    p2 = y2.softmax(dim=1)[:, 1] #输出类别
    p2 = p2.numpy() # no_grad修饰后不需要detach
d2 = (d_test==1).astype(np.int32)
level = np.linspace(0, 1, 20)
TPR, FPR = [], []
for e in level:# 真实类别
    pt = (p2>=e).astype(np.int32)
    tp = np.sum((pt==1)*(d2==1))
    fn = np.sum((pt==0)*(d2==1))
    fp = np.sum((pt==1)*(d2==0))
    tn = np.sum((pt==0)*(d2==0))
    tpr = tp/(tp+fn) 
    fpr = fp/(fp+tn)
    TPR.append(tpr) 
    FPR.append(fpr)

fig = plt.figure(1, figsize=(12, 12), dpi=100)
gs = grid.GridSpec(1, 1)
color = ["#ff0000", "#0000ff"]
ax = fig.add_subplot(gs[0])

ax.plot(FPR, TPR, c="k", label="ROC曲线")
x = np.linspace(0, 1, 1000)
ax.plot(x, x, c="k", linestyle="--", label="完全随机")
x = np.ones_like(FPR) * 0 
ax.fill_between(FPR, x, TPR, color="#000000", alpha=0.2, label="AUC")
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_xlim((0, 1))
ax.set_ylim((0, 1))
ax.legend(loc="lower right")
plt.savefig("导出图像/手写数字.ROC.png")
plt.savefig("导出图像/手写数字.ROC.svg")
print(x_train.shape)
          