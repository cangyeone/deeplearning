import numpy as np 

# 定义测试矩阵
X = np.random.random([100, 4]) + 1 # 测试矩阵 
W = np.random.random([100, 4]) + 1 
# 加减乘除计算，都是逐个元素进行的
Y = X + W # 对应元素进行相加
Y = X - W # 对应元素进行相加
Y = X * W # 对应元素进行相加
Y = X / W # 对应元素进行相加
# 如果A的维度为[N, C]
# 如果W的维度为[N, 1]
# 此时相加相当于在1的维度复制C份，此时与A维度相同
# 加减乘除运算相同
W = np.random.random([100, 1]) + 1 
Y = X + W 
# 如果W第0个维度为1那么相当于在0维度复制N份
W = np.random.random([1, 4]) + 1 
Y = X * W 
# 对于最后一个维度可以简化为一个向量
W = np.random.random([4]) + 1 # 与上一步等价
Y = X * W 

# 如果W[1, 4], X[100, 1]，那么二者加减乘除后
# 相当于W复制100份，X复制4份
# np.newaxis为添加一个新的维度
W = np.arange(4)[np.newaxis, ...] #[1, 4] 
X = np.arange(100)[..., np.newaxis]#[100, 1]
Y = X + Y # y.shape :[100, 4] 
# 反过来是不行的