
import numpy as np 
from sklearn.datasets import load_iris

iris = load_iris() 
data = iris.data # 鸢尾花数据 
d = iris.target  # 鸢尾花标签

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
idx1 = np.arange(4)[:, np.newaxis] + np.zeros([1, 3])
idx2 = np.zeros([4, 1]) + np.arange(3) + 1 
idx1 = idx1.astype(np.int32)# Shape:[4, 3] 
idx2 = idx2.astype(np.int32)# Shape:[4, 3]
data6 = data[idx1, idx2]
# 获取标签为0的所有样本
data7 = data[d==0]

