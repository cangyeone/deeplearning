
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid 
import numpy as np 
import torch 
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['font.size'] = "24"  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 制作数据
np.random.seed(8)
# 数据格式:[样本数量，特征数量]
x = np.random.normal(0, 1, [1000, 1]) 
d = x - 1 + np.random.normal(0, 0.3, [1000, 1]) 

def model(x, w1, w2):
    # 定义模型
    return w1 * x  + w2 
def grad(x, d, w1, w2):
    y = model(x, w1, w2)
    gy = 2 * (y - d)#dloss/dy
    gw1 = np.mean(gy * x ) # dloss/dw1
    gw2 = np.mean(gy ) # dloss/dw2
    # 计算损失函数
    loss = np.mean((y-d)**2) 
    return gw1, gw2, loss  
# 定义初始值
w1, w2 = 0, 0
# 学习率
eta = 0.1 
for step in range(200):
    gw1, gw2, loss = grad(x, d, w1, w2) 
    w1 -= eta * gw1 
    w2 -= eta * gw2 
print(w1, w2)
xp = np.linspace(-5, 5, 1000) 
yp = model(xp, w1, w2)
gs = grid.GridSpec(1, 1) 
fig = plt.figure(1, figsize=(12, 9), dpi=100) 
ax = fig.add_subplot(gs[0])
ax.plot(xp, yp, color="#000000", linestyle="--", label="拟合曲线")

ax.scatter(x[:, 0], d[:, 0], 
            color="#888888", 
            #edgecolor="#000000", 
            marker="o", 
            s=30, 
            label="数据点") 
ax.set_xlabel("x")
ax.set_ylabel("d")
ax.set_xlim((-5, 5))
ax.set_ylim((-5, 5))
ax.grid(True)
ax.legend(loc="upper left")
plt.savefig(f"导出图像/直线拟合.svg")
plt.savefig(f"导出图像/直线拟合.png")
print("处理完毕")


