import numpy as np # 矩阵计算库
import matplotlib.pyplot as plt  
plt.switch_backend("TkAgg")
x = np.linspace(0, 10, 100)
y = np.sin(x) 
plt.plot(x, y, c="k", lw=2)
plt.grid(True)
plt.title("SIN function")
plt.show()# 图像展示出来
