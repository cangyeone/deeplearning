
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid 
import numpy as np 
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['font.size'] = "24"  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 制作数据


def sigmoid(x):
    y = 1/(1+np.exp(-x))
    dy = y * (1 - y)
    return y

def dsigmoid(x):
    y = 1/(1+np.exp(-x))
    dy = y * (1 - y)
    return dy


def cal_loss(a):
    return (a**2-2)**2

x = np.linspace(-7, 7, 1000) 
y = sigmoid(x)
dy = dsigmoid(x)
fig = plt.figure(1, figsize=(12, 9), dpi=100)
gs = grid.GridSpec(1, 1)

ax = fig.add_subplot(gs[0])
ax.grid(True)
bbox = dict(boxstyle="round", fc="0.8")
arrowprops = dict(
    arrowstyle = "->",
    connectionstyle = "angle,angleA=0,angleB=90,rad=10")


ax.plot(x, y, c="k", linestyle="-" , label="$\sigma(x)$")
ax.plot(x, dy, c="k", linestyle="--", label="$\sigma'(x)$")

x = 1.0 
ax.annotate(f"此部分导数正常",
            (x, dsigmoid(x)), xytext=(x-2, 0.4), textcoords='data',
            bbox=bbox, arrowprops=arrowprops)
x = 5
ax.annotate(f"此部分梯度接近0",
            (x, dsigmoid(x)), xytext=(x-2, 0.6), textcoords='data',
            bbox=bbox, arrowprops=arrowprops)
ax.legend(loc="upper left")

plt.savefig("导出图像/Sigmoid和导数.jpg")
plt.savefig("导出图像/Sigmoid和导数.svg")