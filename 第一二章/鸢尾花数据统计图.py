
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid 
import numpy as np 
from sklearn.datasets import load_iris 
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['font.size'] = "16"  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

iris = load_iris() 
fnames = ["花萼长度[cm]", "花萼宽度[cm]", "花瓣长度[cm]", "花瓣宽度[cm]"]
tnames = ["山鸢尾", "变色鸢尾", "维吉尼亚鸢尾"]
colors = ["#ff0000", "#00ff00", "#0000ff"]
data = iris.data 
d = iris.target
fig = plt.figure(1, figsize=(12, 6), dpi=150) 
gs = grid.GridSpec(1, 2) 
ax = fig.add_subplot(gs[0]) 
i, j = 1, 2
for k in range(3):
    ax.scatter(data[d==k, i], data[d==k, j], color=colors[k], marker=f"${k+1}$", label=tnames[k])
ax.legend(loc="upper right")
ax.set_xlabel(fnames[i])
ax.set_ylabel(fnames[j])
ax.set_title("a)", x=0.0, y=1.0, va="bottom", ha="left")
ax = fig.add_subplot(gs[1])
i = 2
for k in range(3):
    x1, b1 = np.histogram(data[d==k, i], bins=8, range=(np.min(data[:, [i, j]]), np.max(data[:, [i, j]])))
    #x2, b2 = np.histogram(data[:, j], bins=8, range=(np.min(data[:, [i, j]]), np.max(data[:, [i, j]])))
    ax.bar(b1[:-1], x1, width=b1[1]-b1[0], color=colors[k], alpha=0.5, label=tnames[k])
    #ax.bar(b2[:-1], x2, width=b2[1]-b2[0], color="#0000ff", alpha=0.5, label=tnames[j])
ax.axvline(2.098, c="k", alpha=0.5, lw=2, linestyle="--")
ax.axvline(4.308, c="k", alpha=0.5, lw=2, linestyle="--")
ax.set_xlabel(fnames[i])
ax.set_ylabel("数量")
ax.legend(loc="upper right")
ax.set_title("b)", x=0.0, y=1.0, va="bottom", ha="left")
plt.savefig("导出图像/鸢尾花统计.jpg")
plt.savefig("导出图像/鸢尾花统计.svg")