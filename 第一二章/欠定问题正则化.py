import numpy as np 
import torch 
from plotconfig import * 

A = torch.tensor([[2, -1], [4, -2]], dtype=torch.float32) 
b = torch.tensor([[1.0], [2.0]], dtype=torch.float32) 

output = []
for alpha in [0.01, 0.1, 0.5, 1.0]:
    collect_x = []
    for itr in range(20):
        # 随机选择数字
        x = torch.randn([2, 1], requires_grad=True) # 产生一个随机数 
        # 设定学习率
        eta = 0.01
        # 开始迭代
        for step in range(1000): # 迭代1000次
            pred_b = A @ x  
            # 使得预测的b与真实的b尽可能的接近
            loss = torch.mean((pred_b-b)**2) + alpha * (x**2).mean()
            loss.backward()# 计算导数
            with torch.no_grad():
                x -= eta * x.grad 
                x.grad.zero_()
        print(alpha, pred_b[:, 0], b[:, 0])
        collect_x.append(x.detach().numpy()[:, 0]) 
    collect_x = np.array(collect_x) 
    output.append(collect_x)
fig = plt.figure(1, figsize=(12, 9), dpi=100)
gs = grid.GridSpec(1, 1)
ax = fig.add_subplot(gs[0])
ax.grid(True)
for i, alpha in enumerate([0.01, 0.1, 0.5, 1.0]):
    collect_x = output[i]
    ax.plot(collect_x[:, 0], c="k", marker=f"${i}$", alpha=0.6, linestyle="-" )
    ax.plot(collect_x[:, 1], c="k", marker=f"${i}$", alpha=0.6, linestyle="--")
for i, alpha in enumerate([0.01, 0.1, 0.5, 1.0]):
    ax.plot([], c="k", marker=f"${i}$", alpha=0.6, linestyle="-" , label=rf"$x_1,\alpha={alpha:.2f}$")
for i, alpha in enumerate([0.01, 0.1, 0.5, 1.0]):
    ax.plot([], c="k", marker=f"${i}$", alpha=0.6, linestyle="--", label=rf"$x_2,\alpha={alpha:.2f}$")
ax.legend(loc="lower right", ncol=2, fontsize=12)
ax.set_xlabel("实验次数")
ax.set_ylim((-2.5, 2.5))
plt.savefig("导出图像/欠定问题正则化.svg")
plt.savefig("导出图像/欠定问题正则化.png")
plt.show()
