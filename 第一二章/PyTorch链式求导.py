import torch 

# 对于需要求导的量需要定义requries_grad=True
x = torch.ones([1], requires_grad=True)
y1 = x ** 2 
y2 = torch.sin(y1) 

# dy2/dx 
y2.backward() # 从y2开始计算
# 通过链式法则计算变量x导数
print(x.grad) # dy2/dx 
y1.backward()

# 由于之前已经执行过反向传播了，因此不能再次计算
try:
    y1.backward()
except:
    print("程序运行错误")

# 需要重新进行计算以进行求导 
y1 = x ** 2 
y2 = torch.sin(y1) 
# 从y1开始计算的话就是dy1/dx 
y1.backward()
# 此时梯度与理论梯度值不相等
print(x.grad)

# 多次计算的过程grad是累加的
# 因此在执行新的计算时梯度应当置零
x.grad.zero_() # 带_的函数为对原位处理，即对相应内存进行置0
y1 = x ** 2 
y2 = torch.sin(y1) 
# 从y1开始计算的话就是dy1/dx 
y1.backward()
# 此时梯度便正常了
print(x.grad)
