
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid 
import numpy as np 
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['font.size'] = "24"  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 制作数据


def sqrt(x): # 开根号
    eta = 0.1 # 定义学习率 
    a = 1 # 定义初始值
    for step in range(20):
        g = 2 * (a**2-x) * 2 * a 
        a -= eta * g 
    return a
print(sqrt(2))


def cal_loss(a):
    return (a**2-2)**2

a = np.linspace(-2, 2, 1000) 
loss = (a**2-2)**2 
fig = plt.figure(1, figsize=(12, 9), dpi=100)
gs = grid.GridSpec(1, 1)

ax = fig.add_subplot(gs[0])
ax.grid(True)
bbox = dict(boxstyle="round", fc="0.8")
arrowprops = dict(
    arrowstyle = "->",
    connectionstyle = "angle,angleA=0,angleB=90,rad=10")


ax.plot(a, loss, c="k")

x = 1.0 
ax.annotate(f"初始值1:{x:.1f}",
            (x, cal_loss(x)), xytext=(0.0, 1.5), textcoords='data',
            bbox=bbox, arrowprops=arrowprops)
x = -1.0 
ax.annotate(f"初始值2:{x:.1f}",
            (x, cal_loss(x)), xytext=(-0.5, 2.5), textcoords='data',
            bbox=bbox, arrowprops=arrowprops)


x = 1.414
ax.annotate(f"收敛值1:{x:.3f}",
            (1.4142, 0.0), xytext=(0.2, 0.1), textcoords='data',
            bbox=bbox, arrowprops=arrowprops)
x = -1.414
ax.annotate(f"收敛值2:{x:.3f}",
            (-1.4142, 0.0), xytext=(-1.0, 0.1), textcoords='data',
            bbox=bbox, arrowprops=arrowprops)
x = 1
ax.scatter([x, -x], [cal_loss(x), cal_loss(x)], c="k", marker="o")
x = 1.41423
ax.scatter([x, -x], [0, 0], c="k", marker="o")
ax.set_xlabel("求解$\sqrt{2}$")
ax.set_ylabel("loss")

plt.savefig("导出图像/求根号.png")
plt.savefig("导出图像/求根号.svg")