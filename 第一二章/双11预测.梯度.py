
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid 
import numpy as np 
import torch 
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['font.size'] = "24"  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 制作数据
data = [[2009, 0.5], [2010, 9.36], [2011, 33.6], 
        [2012, 191], [2013, 350], [2014, 571], 
        [2015, 912], [2016, 1207], [2017, 1682], 
        [2018, 2135], [2019, 2684], [2020, 4982], 
        [2021, 4100]] 





data1 = np.array(data) 
# 2020年数据异常需要剔除
data.pop(11)
data2 = np.array(data) 
out = []

x, d = data2[:, 0], data2[:, 1] # x和d
# 数据标准化
x = (x-2009) / 10 
d = d / 5000 


def model(x, w1, w2):
    # 定义模型
    return w1 * x ** 2 + w2 * x
def grad(x, d, w1, w2):
    y = model(x, w1, w2)
    gy = 2 * (y - d)#dloss/dy
    gw1 = np.mean(gy * x ** 2) # dloss/dw1
    gw2 = np.mean(gy * x) # dloss/dw2
    # 计算损失函数
    loss = np.mean((y-d)**2) 
    return gw1, gw2, loss  
# 定义初始值
w1, w2 = 0, 0
# 学习率
eta = 0.3 
for step in range(200):
    gw1, gw2, loss = grad(x, d, w1, w2) 
    w1 -= eta * gw1 
    w2 -= eta * gw2 
print(w1, w2)
x = np.linspace(0, 1.3, 1000) 
y = model(x, w1, w2)
x = x * 10 + 2009 
y = y * 5000 
gs = grid.GridSpec(1, 1) 
fig = plt.figure(1, figsize=(12, 9), dpi=100) 
ax = fig.add_subplot(gs[0])
ax.plot(x, y, color="#666666", linestyle="--", label="优化结果")

ax.scatter(data1[:, 0], data1[:, 1], color="#ffffff", edgecolor="#000000", marker="o", s=200) 
ax.set_xlabel("年份")
ax.set_ylabel("销售额（亿）")
ax.set_xlim((2009, 2022))
ax.grid(True)
ax.legend(loc="upper left")
plt.savefig(f"导出图像/双11预测.梯度.jpg")
plt.savefig(f"导出图像/双11预测.梯度.pdf")

print(data) 

