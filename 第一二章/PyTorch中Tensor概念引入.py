import numpy as np 
import torch  

# 定义一个NumPy中的ndarray 
x_np = np.zeros([2])
# x_np中的复制一份转换为Tensor
x_torch = torch.tensor(x_np)
# 对矩阵中的数据进行更改不影响原始数据
x_torch[0] = 1.0 
print(x_np) # 依然是[0, 0]

# x_np与x_torch共享内存
x_torch = torch.from_numpy(x_np)
# 对矩阵中的数据进行更改影响原始数据
x_torch[0] = 1.0 
print(x_np) # 此时是[1, 0]

# Tensor本身可以由NumPy类似的方法构建
y_torch = torch.ones([2]) 

# Tensor可以通过.numpy()方法转换为ndarray 
y_np = y_torch.numpy() 
print(type(y_np))#ndarray