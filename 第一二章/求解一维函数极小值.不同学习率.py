
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid 
import numpy as np 
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['font.size'] = "24"  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 制作数据


def f(x): # 定义函数
    return x**2 - x
def grad(x): # 定义导数/梯度
    return 2 * x - 1

datas = []
etas = [0.01, 0.1, 1.1]
for eta in etas:
    x = 0 # 定义初始值
    data = []
    data.append(f(x))
    for step in range(10):
        g = grad(x)
        x -= eta * g 
        data.append(f(x))
    datas.append(data)
gs = grid.GridSpec(1, 1) 
fig = plt.figure(1, figsize=(12, 9), dpi=100) 
ax = fig.add_subplot(gs[0])
ax.grid(True)
for i in range(3):
    ax.plot(datas[i], marker=f"${i}$", label=f"{etas[i]:.3f}")
ax.legend(loc="upper right")
ax.set_xlabel("迭代次数")
ax.set_ylabel("f(x)")
ax.set_ylim((-0.3, 1.5))

plt.savefig(f"导出图像/一元函数不同eta.png")
plt.savefig(f"导出图像/一元函数不同eta.svg")

print(data) 

