
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


def model(x, w1, w2):
    # 定义模型
    return w1 * x ** 2 + w2 * x
def cal_loss(x, d, w1, w2):
    y = model(x, w1, w2)
    # 计算损失函数
    loss = np.mean((y-d)**2) 
    return loss  


data1 = np.array(data) 
# 2020年数据异常需要剔除
data.pop(11)
data2 = np.array(data) 
out = []
for tdata in [data1, data2]:
    w1 = np.linspace(0, 3, 1001) # 构建搜索网格
    w2 = np.linspace(0, 3, 1001) # 搜索范围
    x, d = tdata[:, 0], tdata[:, 1] # x和d
    # 数据标准化
    x = (x-2009) / 10 
    d = d / 5000 

    best_w1 = 0 
    best_w2 = 0
    min_loss = cal_loss(x, d, best_w1, best_w2)
    for itr_w1 in w1:
        for itr_w2 in w2:
            loss = cal_loss(x, d, itr_w1, itr_w2) 
            if loss < min_loss:
                min_loss = loss 
                best_w1 = itr_w1 
                best_w2 = itr_w2 
    print(f"最佳参数{best_w1:.3f},{best_w2:.3f}。\n2022年预测{model((2022-2009)/10, best_w1, best_w2)*5000}")

    x = np.linspace(0, 1.3, 1000) 
    y = model(x, best_w1, best_w2)

    x = x * 10 + 2009 
    y = y * 5000 
    out.append(y) 
gs = grid.GridSpec(1, 1) 
fig = plt.figure(1, figsize=(12, 9), dpi=100) 
ax = fig.add_subplot(gs[0])
ax.plot(x, out[0], color="#666666", linestyle="--", label="带异常数据训练模型")
ax.plot(x, out[1], color="#333333", linestyle="-", label="清洗后数据训练模型")

ax.scatter(data1[:, 0], data1[:, 1], color="#ffffff", edgecolor="#000000", marker="o", s=200) 
ax.set_xlabel("年份")
ax.set_ylabel("销售额（亿）")
ax.set_xlim((2009, 2022))
ax.grid(True)
ax.legend(loc="upper left")
plt.savefig(f"导出图像/双11预测.png")
plt.savefig(f"导出图像/双11预测.svg")

print(data) 

