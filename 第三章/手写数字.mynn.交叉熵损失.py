import numpy as np 
from plotconfig import * 

file_ = np.load("data/mnist.npz")
x_train = file_["x_train"]
d_train = file_["y_train"]
x_test = file_["x_test"]
d_test = file_["y_test"]
x_train = x_train.astype(np.float32)/255 
x_test = x_test.astype(np.float32)/255 

# 使用自行编写的深度学习库完成机器学习问题
import mynn.nn as nn 
import mynn 
import mynn.nn.functional as F 
x1 = x_train.reshape([-1, 784])
x1 = mynn.Tensor(x1) # 转换为Tensor类型
d1 = d_train.reshape([-1])
d1 = mynn.Tensor(d1) # 转换为Tensor类型

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
        self.layer1 = nn.Linear(784, 32) 
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.layer1(x)
        h = self.relu(x) 
        y = self.layer2(h)
        return y 

model = Model() 
pars = model.parameter()
#pdict = mynn.load("mnist.mynn")
#model.load_state_dict(pdict)

# 超参数定义
eta = 0.001 # 学习率
alpha = 0.001 # 正则化系数
batch_size = 10
n_epoch = 50 
optim = mynn.optim.SGD(pars)
lossfn = open("temp/losssgd.txt", "w", encoding="utf-8")
x2 = mynn.Tensor(x_test.reshape([-1, 784]))[:100]
d2 = d_test[:100]
for e in range(n_epoch):
    for step in range(len(x1)//batch_size):
        idx = np.random.randint(0, len(x1), [batch_size])
        x = x1[idx] 
        d = d1[idx] 
        y = model(x)
        loss = F.cross_entropy(y, d)
        loss.backward() 
        optim.step()
        optim.zero_grad()
        
        y2 = model(x2)
        p2 = y2.data.argmax(axis=1) #输出类别
        #print(p2.shape, d2.shape)
        lossfn.write(f"{e},{step},{loss.data},{np.mean(p2==d2)}\n")
    #print(e, step, loss, np.max(np.abs(pars[0].data)))
    mynn.save(model.state_dict(), "mnist.mynn")
    x2 = mynn.Tensor(x_test.reshape([-1, 784]))
    y2 = model(x2)
    p2 = y2.data.argmax(axis=1) #输出类别
    print(p2, x_test.reshape([-1, 784]).shape, x2.data.shape, p2.shape)
    print(f"预测准确度{np.mean(p2==d_test)}")
#fig = plt.figure(1, figsize=(12, 9), dpi=100)
#gs = grid.GridSpec(3, 4)
#for i in range(3):
#    for j in range(4):
#        x = x_test[i*4+j]
#        d = d_test[i*4+j]
#        p = p2[i*4+j]
#        ax = fig.add_subplot(gs[i, j])
#        ax.matshow(x, cmap=plt.get_cmap("Greys"))
#        ax.set_xticks(())
#        ax.set_yticks(())
#        ax.set_xlabel(f"真实类别:{d},预测类别:{p}")
#plt.savefig("导出图像/手写数字.torch.jpg")
#plt.savefig("导出图像/手写数字.torch.pdf")
#print(x_train.shape)
#          