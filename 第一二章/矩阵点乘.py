import matplotlib.pyplot as plt
import matplotlib.gridspec as grid 
import numpy as np 
import torch 
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['font.size'] = "24"  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

X = np.zeros([300, 2]) 
X[:100, 0] = np.linspace(-1, 1, 100) 
X[:100, 1] = X[:100, 0] ** 2 - 1 

X[100:200, 0] = np.cos(np.linspace(0, np.pi*2, 100))
X[100:200, 1] = np.sin(np.linspace(0, np.pi*2, 100))

X[200:300, 0] = np.linspace(-1, 1, 100)
X[200:300, 1] = np.linspace(-1, 1, 100)

# 定义矩阵
o = np.pi / 4 
E = np.array([[np.cos(o), -np.sin(o)], 
              [np.sin(o), np.cos(o)]])
A = np.diag([2.0, 0.5]) # 对角矩阵
b = np.array([1, 1])
# 使用NumPy矩阵点乘和加法 
Y = X @ (E @ A) + b 
# 使用PyTorch运算也可以
X_torch = torch.tensor(X, dtype=torch.float32) 
E_torch = torch.tensor(E, dtype=torch.float32) 
A_torch = torch.tensor(A, dtype=torch.float32) 
b_torch = torch.tensor(b, dtype=torch.float32) 
Y_torch = X_torch @ (E_torch @ A_torch) + b_torch 
Y = Y_torch.cpu().numpy()


gs = grid.GridSpec(1, 1) 
fig = plt.figure(1, figsize=(12, 12), dpi=100) 
ax = fig.add_subplot(gs[0])
ax.scatter(X[:, 0], X[:, 1], c="r", label="原始数据")
ax.scatter(Y[:, 0], Y[:, 1], c="b", label="变换后数据")
for a1, a2 in zip(X, Y):
    ax.plot([a1[0], a2[0]], [a1[1], a2[1]], c="k", alpha=0.1)
ax.legend(loc="upper right")
ax.axis("equal")
ax.set_xlabel("x1/y1")
ax.set_ylabel("x2/y2")
ax.set_title("$Y=X\cdot W + b$")
plt.savefig("导出图像/矩阵点乘-仿射变换.jpg")
plt.savefig("导出图像/矩阵点乘-仿射变换.svg")