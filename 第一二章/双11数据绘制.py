
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid 
import numpy as np 
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
# 2020年数据异常需要剔除
# 转换为ndarray格式
data = np.array(data) 

gs = grid.GridSpec(1, 1) 
fig = plt.figure(1, figsize=(12, 9), dpi=100) 
ax = fig.add_subplot(gs[0])
ax.plot(data[:, 0], data[:, 1], color="#666666", linestyle="--") 
ax.scatter(data[:, 0], data[:, 1], color="#ffffff", edgecolor="#000000", marker="o", s=200) 
ax.set_xlabel("年份")
ax.set_ylabel("销售额（亿）")
ax.set_xlim((2009, 2022))
ax.grid(True)

plt.savefig(f"导出图像/双11.png")
plt.savefig(f"导出图像/双11.svg")

print(data) 

