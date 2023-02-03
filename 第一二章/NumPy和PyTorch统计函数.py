from matplotlib.pyplot import axis
import numpy as np 
import torch  

x_np = np.random.normal(0, 1, [10, 10, 3])
x_torch = torch.randn([10, 10, 3])

# 计算某一个维度均值有多种方式
mu_np = x_np.mean(axis=1) 
mu_np = np.mean(x_np, axis=1) 
print(mu_np.shape)#(10, 3)
mu_np = x_np.mean(axis=1, keepdims=True)
mu_np = np.mean(x_np, axis=1, keepdims=True)
print(mu_np.shape)#(10, 1, 3)

# PyTorch与NumPy函数类似，但是有细微区别
mu_torch = x_torch.mean(dim=1)
mu_torch = torch.mean(x_torch, dim=1)
print(mu_torch.shape)#torch.Size([10, 3])
mu_torch = x_torch.std(dim=1, keepdim=True)
mu_torch = torch.std(x_torch, dim=1, keepdim=True)
print(mu_torch.shape)#torch.Size([10, 1, 3])