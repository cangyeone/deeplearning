# 计算机视觉库
from torchvision.models import resnet50, mobilenet_v2 
import torch 
import time 

model1 = resnet50() # 2015年残差网络
model2 = mobilenet_v2()# 卷积速度优化
x = torch.randn([1, 3, 224, 224], dtype=torch.float32)
for i in range(10):
    t1 = time.perf_counter()
    y = model2(x)
    t2 = time.perf_counter()
    print(f"{t2-t1:.3f}")
torch.save(model1.state_dict(), "resnet.pth")
torch.save(model2.state_dict(), "mobilenetv2.pth")