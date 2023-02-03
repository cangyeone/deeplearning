
from plotconfig import * 
import numpy as np 

std = 0.5 
x1 = np.random.normal(1, std, [200, 2]) 
x1[:100] = np.random.normal(-1, std, [100, 2]) 
d1 = np.zeros([200]) 
d1[:100] = 1

x2 = np.random.normal(1, std, [200, 2]) 
x2[50:100] = np.random.normal(-1, std, [50, 2]) 
x2[100:150] = np.random.normal(0, std, [50, 2]) + np.array([-1, 1])
x2[150:200] = np.random.normal(0, std, [50, 2]) + np.array([1, -1])
d2 = np.zeros([200]) 
d2[:100] = 1


fig = plt.figure(1, figsize=(12, 6), dpi=100) 
gs = grid.GridSpec(1, 2) 

ax1 = fig.add_subplot(gs[0]) 
ax2 = fig.add_subplot(gs[1]) 

for i in range(2):
    ax1.scatter(x1[d1==i, 0], x1[d1==i, 1], c="k", marker=f"${i+1}$", label=f"类别{i+1}") 

for i in range(2):
    ax2.scatter(x2[d2==i, 0], x2[d2==i, 1], c="k", marker=f"${i+1}$", label=f"类别{i+1}") 

x = np.linspace(-3, 3, 1000) 
y1 = x 
y2 = 0.1 * x 

ax1.plot(x, -y1, c="k", linestyle="-", label="分割曲线/面")
ax2.plot(x, y2, c="k", linestyle="--", label="不存在分割曲面")

ax1.set_title("a)", x=0.0, y=1.0, va="bottom", ha="left")
ax2.set_title("b)", x=0.0, y=1.0, va="bottom", ha="left")
for ax in [ax1, ax2]:
    ax.legend(loc="upper right")
    ax.set_xlim((-3, 3)) 
    ax.set_ylim((-3, 3))

plt.savefig("导出图像/线性可分与不可分.svg")
plt.savefig("导出图像/线性可分与不可分.png")