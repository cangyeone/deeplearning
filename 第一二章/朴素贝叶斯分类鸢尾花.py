import numpy as np 
from sklearn.datasets import load_iris 
iris = load_iris() 

data = iris.data 
d = iris.target

# 正态分布
def norm(x, mu, std):
    # 正态分布形式
    # x矩阵格式[样本数量，属性数量]
    # mu,std均值标准差，格式[属性数量]
    # 可以回忆之前矩阵运算部分
    return 1/np.sqrt(2*np.pi)/std * \
        np.exp(-(x-mu)**2/std**2/2)
# 将三类鸢尾花数据分开统计
mu = [] 
std = []
num = []
for i in range(3):
    x = data[d==i]# 获取相应类别数据
    # 统计i类鸢尾花均值和标准差
    mu.append(np.mean(x, axis=0))
    std.append(np.std(x, axis=0))
    num.append(len(x))

# 计算三类鸢尾花类别概率
x_test = data # 用于测试的数据
prob = []
for i in range(3):
    patt = norm(x_test, mu[i], std[i])# 概率
    logp = np.log(patt) # 防止出现数值问题
    #每个属性乘法变为log后加法
    p = np.sum(logp, axis=1) 
    p += np.log(num[i]/np.sum(num))# 加入先验
    prob.append(np.exp(p)) 
prob = np.stack(prob).T # [样本数量，每类归一化前概率]
prob /= np.sum(prob, axis=1, keepdims=True) # 归一化 
pred_class = np.argmax(prob, axis=1)
print(pred_class)
print("准确率", np.mean(pred_class==d))


