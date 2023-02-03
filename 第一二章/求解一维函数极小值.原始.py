
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid 
import numpy as np 
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['font.size'] = "24"  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 制作数据


def f(x): # 定义函数
    return x**2 - x
def grad(x): # 定义导数/梯度
    return 2 * x - 1

x = 0 # 定义初始值
eta = 0.1 # 定义学习率 
for step in range(20):
    g = grad(x)
    x -= eta * g 
    print(f"{step:03}:f({x:.3f})={f(x):.3f}")
