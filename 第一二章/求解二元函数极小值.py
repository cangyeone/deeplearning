

def f(x1, x2): # 函数
    return x1 ** 2 + x2 ** 2 + 2 * x1 + x2 

def grad(x1, x2): # 函数梯度
    return 2 * x1 + 2, 2 * x2 + 1 

# 设定初始值
x1, x2 = 0, 0 
# 设定学习率0.1 
eta = 0.1 
# 开始迭代，最多迭代20次
for step in range(20):
    g1, g2 = grad(x1, x2)
    x1 -= eta * g1 
    x2 -= eta * g2 
    print(f"f({x1:.2f},{x2:.2f})={f(x1,x2):.3f}")