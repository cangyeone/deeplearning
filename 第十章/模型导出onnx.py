import torch 
import torch.nn as nn 

model = nn.Sequential(
    nn.Conv2d(1, 16, 3, stride=1, padding=1), 
    nn.ReLU(), 
    nn.AvgPool2d(2, 2),
    nn.Conv2d(16, 32, 3, stride=1, padding=1), 
    nn.ReLU(), 
    nn.AvgPool2d(2, 2),
    nn.Flatten(), 
    nn.Linear(32*7*7, 10)
)

model.eval()
#model.fuse_model()
input_names = [ "image" ] # 定义输入名称
output_names = [ "class" ]# 定义输出名称
# 模拟输入
dummy_input = torch.randn([1, 1, 28, 28])
# 导出模型
torch.onnx.export(
 model, dummy_input, 
 "mnist.onnx", # 模型名称
 verbose=True, 
 input_names=input_names, 
 output_names=output_names)