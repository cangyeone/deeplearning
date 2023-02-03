import numpy as np 
import torch 

A = torch.tensor([[1, -1.5], [2.0, 0.5]], dtype=torch.float32) 
b = torch.tensor([[1.0], [-1.0]], dtype=torch.float32) 
# 可导的
x = torch.randn([2, 1], requires_grad=True) # 产生一个随机数 
# 设定学习率
eta = 0.3 
# 开始迭代
for step in range(100): # 迭代100次
    pred_b = A @ x  
    # 使得预测的b与真实的b尽可能的接近
    loss = torch.sum((pred_b-b)**2) 
    loss.backward()# 计算导数
    with torch.no_grad():
        x -= eta * x.grad 
        x.grad.zero_()
print(loss)
print(x)
# 使用矩阵求逆方式进行计算
x_new = torch.linalg.inv(A) @ b
print(x_new)