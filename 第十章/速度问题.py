import numpy as np # 矩阵运算使用
import time        # 计时
# 构建两个矩阵
a = np.random.random([4000, 4000])
b = np.random.random([4000, 4000])
# 数值精度
# float64双精度浮点
# float32单精度浮点
# float16半精度浮点
# 低bit计算
# 边缘设备：内存小，浮点计算慢
a = a.astype(np.float16)
b = b.astype(np.float16)
for i in range(10):
    t1 = time.perf_counter() 
    c = a @ b # 矩阵乘法，算法复杂度n^3
    t2 = time.perf_counter()
    print(f"{t2-t1:.3f}")