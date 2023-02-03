import torch 
import torch.nn as nn 
import time 
# 产生随机测试数据
x = torch.randn([32, 64, 300, 300]) 
# 定义卷积网络
# 逐层卷积
cnn1 = nn.Conv2d(64, 64, 3, groups=64) 
# 逐点卷积 
cnn2 = nn.Conv2d(64, 128, 1, groups=1)

# MobileNet V1~V3 
cnn1.eval() # 推断模型
cnn2.eval()
for i in range(10):
    t1 = time.perf_counter() 
    x = cnn1(x)
    y = cnn2(x)
    t2 = time.perf_counter()
    print(f"{t2-t1:.3f}")

# 5×5:6秒左右
# 3×3:2.8秒左右
# 3×3+分2组:1.8秒左右
# 3×3+分64组:1.7秒左右