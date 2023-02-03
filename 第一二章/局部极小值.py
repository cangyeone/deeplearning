
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid 
import numpy as np 
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['font.size'] = "24"  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 制作数据
def f(x):
    return -np.cos(5*x) * 2 + x ** 2 

x = np.linspace(-2*np.pi, 2*np.pi, 1000) 
y = f(x)
fig = plt.figure(1, figsize=(12, 9), dpi=100)
gs = grid.GridSpec(1, 1)

ax = fig.add_subplot(gs[0])
ax.grid(True)
bbox = dict(boxstyle="round", fc="0.8")
arrowprops = dict(
    arrowstyle = "->",
    connectionstyle = "angle,angleA=0,angleB=90,rad=10")


ax.plot(x, y, c="k")

x = 0.0 
ax.annotate(f"全局极小值:{x:.1f}",
            (x, f(x)), xytext=(x, f(x)-4.5), textcoords='data',
            bbox=bbox, arrowprops=arrowprops)
ax.scatter([x], [f(x)], c="k", marker="o")
for i in range(1, 3):
    x = -2*np.pi/5 * i  
    ax.annotate(f"局部极小值",
                (x, f(x)), xytext=(-4, 35), textcoords='data',
                bbox=bbox, arrowprops=arrowprops)
    ax.scatter([x], [f(x)], c="k", marker="o")
    x = 2*np.pi/5 * i  
    ax.annotate(f"局部极小值",
                (x, f(x)), xytext=(1.5, 35), textcoords='data',
                bbox=bbox, arrowprops=arrowprops)
    ax.scatter([x], [f(x)], c="k", marker="o")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_ylim((-10, 40))

plt.savefig("导出图像/局部极小值.png")
plt.savefig("导出图像/局部极小值.svg")
plt.show()