import numpy as np 
import torch 
from plotconfig import * 

A = torch.tensor([[2, -1], [4, -2]], dtype=torch.float32) 
b = torch.tensor([[1.0], [2.0]], dtype=torch.float32) 

collect_x = []
for itr in range(20):
    # 随机选择数字
    x = torch.randn([2, 1], requires_grad=True) # 产生一个随机数 
    # 设定学习率
    eta = 0.01
    alpha = 0.0 
    # 开始迭代
    for step in range(100): # 迭代100次
        pred_b = A @ x  
        # 使得预测的b与真实的b尽可能的接近
        loss = torch.sum((pred_b-b)**2) + alpha * (x**2).sum()
        loss.backward()# 计算导数
        with torch.no_grad():
            x -= eta * x.grad 
            x.grad.zero_()
    collect_x.append(x.detach().numpy()[:, 0]) 
collect_x = np.array(collect_x) 
print(collect_x)
print(collect_x.shape)
fig = plt.figure(1, figsize=(12, 9), dpi=100)
gs = grid.GridSpec(1, 1)
ax = fig.add_subplot(gs[0])
ax.grid(True)
ax.plot(collect_x[:, 0], c="k", linestyle="-" , label="$x_1$")
ax.plot(collect_x[:, 1], c="k", linestyle="--", label="$x_2$")
ax.legend()
ax.set_xlabel("实验次数")
plt.savefig("导出图像/欠定问题.svg")
plt.savefig("导出图像/欠定问题.png")
plt.show()
