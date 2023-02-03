
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid 
import numpy as np 
from sklearn.datasets import load_iris
import xarray 
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['font.size'] = "16"  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

iris = load_iris() 
fnames = ["花萼长度[cm]", "花萼宽度[cm]", "花瓣长度[cm]", "花瓣宽度[cm]"]
tnames = ["山鸢尾", "变色鸢尾", "维吉尼亚鸢尾"]

data = iris.data # 鸢尾花数据 
d = iris.target  # 鸢尾花标签

def normal(length, mu, std):
    # 正态分布形态
    return 1/np.sqrt(2*np.pi)/std * np.exp(-(length-mu)**2/std**2/2)
idx = 1 
x3  = data[:, idx] # 花瓣长度
mu = [np.mean(x3[d==i]) for i in range(3)] # 统计每类均值
std = [np.std(x3[d==i]) for i in range(3)] # 统计每类标准差

predid = []
for iris1 in x3:
    probs = []
    for k in range(3):
        prob = normal(iris1, mu[k], std[k])#计算概率
        probs.append(prob) 
    predid.append(np.argmax(probs)) 
predid = np.array(predid)
print(f"准确度为{np.mean(predid==d)}, 错误数量{np.sum(predid!=d)}")


colors = ["#ff0000", "#00ff00", "#0000ff"]


fig = plt.figure(1, figsize=(12, 9), dpi=150) 
gs = grid.GridSpec(1, 1) 

ax = fig.add_subplot(gs[0])
i = 1
t = np.linspace(1, 7, 1000)
for k in range(3):
    ax.plot(t, normal(t, mu[k], std[k]), color=colors[k], lw=1.5, label=f"$p_{k+1}(x)$")
    ax.hist(x3[d==k], bins=8, color=colors[k], alpha=0.5, label=tnames[k], density=True)

#ax.axvline(2.098, c="k", alpha=0.5, lw=2, linestyle="--")
#ax.axvline(4.308, c="k", alpha=0.5, lw=2, linestyle="--")
ax.set_xlabel(fnames[i])
ax.set_ylabel("数量")
ax.legend(loc="upper right")
plt.savefig(f"导出图像/鸢尾花模型{idx}.jpg")
plt.savefig(f"导出图像/鸢尾花模型{idx}.svg")