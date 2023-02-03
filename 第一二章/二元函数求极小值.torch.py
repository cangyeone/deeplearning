import torch 

def model(x):# 定义需要求极小值的函数
    y = (x ** 2).sum() + 2 * x[0] + x[1] 
    return y 

# 定义初始值
x = torch.zeros([2], requires_grad=True) # 向量[x1,x2]
# 定义学习率
eta = 0.1 

for step in range(30):
    f = model(x) 
    f.backward()# 计算导数
    # 本部分不是计算图部分，不需要计算梯度
    with torch.no_grad():
        x -= eta * x.grad # 梯度下降法
        x.grad.zero_()# 梯度不累加
# 由于x为计算图中的节点
# 因此需要使用detach()将其分离出来
# detach()相当于在计算图中进行了截断
x_np = x.detach().numpy()
print(x_np)

