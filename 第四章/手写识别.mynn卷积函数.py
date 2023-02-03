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
        self.layer1 = nn.Conv2d(1, 16, 3, 2, 1) 
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Conv2d(16, 32, 3, 2, 1) 
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(7*7*32, 10)

    def forward(self, x):
        x = x.reshape([-1, 1, 28, 28])
        x = self.layer1(x)
        x = self.relu1(x) 
        x = self.layer2(x)
        x = self.relu2(x) 
        x = x.reshape([-1, 7*7*32]) 
        y = self.layer3(x)

        return y 

model = Model() 
pars = model.parameter()
pdict = mynn.load("ckpt/mnist.cnn.mynn")
model.load_state_dict(pdict)

# 超参数定义
eta = 0.001 # 学习率
alpha = 0.001 # 正则化系数
batch_size = 2
n_epoch = 50 
optim = mynn.optim.Adam(pars)
x2 = mynn.Tensor(x_test.reshape([-1, 784])[:300])
d2 = d_test[:300]
for e in range(n_epoch):
    for step in range(len(x1)//batch_size):
        x = x1[step*batch_size:(step+1)*batch_size] 
        d = d1[step*batch_size:(step+1)*batch_size] 
        y = model(x)
        loss = F.cross_entropy(y, d)
        loss.backward() 
        optim.step()
        optim.zero_grad()
        if step % 50 ==0:
            #print(e, step, loss, np.max(np.abs(pars[0].data)))
            mynn.save(model.state_dict(), "ckpt/mnist.cnn.mynn")
            y2 = model(x2)
            p2 = y2.data.argmax(axis=1) #输出类别
            print(f"{loss}, 预测准确度{np.mean(p2==d2)}")
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
          