
import numpy as np 
from sklearn.datasets import load_iris
import torch 
iris = load_iris() 
data = iris.data # 鸢尾花数据 
d = iris.target  # 鸢尾花标签

# 需要将ndarray转换到torch.Tensor的类型
## 数据一般为单精度浮点
data = torch.tensor(data, dtype=torch.float32) 
## 标签或者索引一般为长整形
d = torch.tensor(d, dtype=torch.long) 
# 可以在GPU上执行
device = torch.device("cuda") 
## 将数据拷贝到GPU上
data = data.to(device) 
d = d.to(device)
# 获取前10个样本
data1 = data[:10] 
# 获取第10-20个样本
data2 = data[10:20] # 10<=idx<20
# 获取第一个属性
data3 = data[:, 1] 
# 获取0-3属性
data4 = data[:, 0:3] 
# 获取1,2,3,7,8,9个样本
data5 = data[[1,2,3,7,8,9]]
# 获取0,1,2,3号样本的1,2,3列属性
## 方式1:先获取样本，再获取属性
data6 = data[[0,1,2,3]][:, [1, 2, 3]]
## 方式2:按索引查找，需要idx1,idx2索引即样本和属性索引
## 这种方式是比较难以理解的
idx1 = torch.arange(4)[:, None] + torch.zeros([1, 3])
idx2 = torch.zeros([4, 1]) + np.arange(3) + 1 
idx1 = idx1.long().to(device)# 索引为长整形，Shape:[4, 3] 
idx2 = idx2.long().to(device)# 索引为长整形，Shape:[4, 3]
data6 = data[idx1, idx2]
# 获取标签为0的所有样本
data7 = data[d==0]

# 可以将Tensor类型转换为ndarray类型
# 在此之前需要首先将数据从GPU拷贝到CPU上
data7 = data7.cpu().numpy()
print(data7)

