import torch 
import torch.nn as nn 
import torch.nn.functional as F

class ConvBNReLU(nn.Sequential):
    """
    三个层在计算过程中应当进行融合
    使用ReLU作为激活函数可以限制
    数值范围，从而有利于量化处理。
    """
    def __init__(self, n_in, n_out, 
                 kernel_size=3, stride=1, 
                 groups=1, norm_layer=nn.BatchNorm2d):
        # padding为same时两边添加(K-1)/2个0
        padding = (kernel_size - 1) // 2
        # 本层构建三个层，即0：卷积，1：批标准化，2：ReLU
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(n_in, n_out, kernel_size, 
                      stride, padding, groups=groups, 
                      bias=False),
            norm_layer(n_out),
            nn.ReLU(inplace=True)
        )
class InvertedResidual(nn.Module):
    """
    本个模块为MobileNetV2中的可分离卷积层
    中间带有扩张部分，如图10-2所示
    """
    def __init__(self, n_in, n_out, 
                 stride, expand_ratio, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.stride = stride
        # 隐藏层需要进行特征拓张，以防止信息损失
        hidden_dim = int(round(n_in * expand_ratio))
        # 当输出和输出维度相同时，使用残差结构
        self.use_res = self.stride == 1 and n_in == n_out
        # 构建多层
        layers = []
        if expand_ratio != 1:
            # 逐点卷积，增加通道数
            layers.append(
                ConvBNReLU(n_in, hidden_dim, kernel_size=1, 
                            norm_layer=norm_layer))
        layers.extend([
            # 逐层卷积，提取特征。当groups=输入通道数时为逐层卷积
            ConvBNReLU(
                hidden_dim, hidden_dim, 
                stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # 逐点卷积，本层不加激活函数
            nn.Conv2d(hidden_dim, n_out, 1, 1, 0, bias=False),
            norm_layer(n_out),
        ])
        # 定义多个层
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
class QInvertedResidual(InvertedResidual):
    """量化模型修改"""
    def __init__(self, *args, **kwargs):
        super(QInvertedResidual, self).__init__(*args, **kwargs)
        # 量化模型应当使用量化计算方法
        self.skip_add = nn.quantized.FloatFunctional()
    def forward(self, x):
        if self.use_res:
            # 量化加法
            return self.skip_add.add(x, self.conv(x))
            #return x + self.conv(x)
        else:
            return self.conv(x)
    def fuse_model(self):
        # 模型融合
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) == nn.Conv2d:
                # 将本个模块最后的卷积层和BN层融合
                fuse_modules(
                    self.conv, 
                    [str(idx), str(idx + 1)], inplace=True)
from torch.quantization import QuantStub, DeQuantStub
class Model(nn.Module):
    """
    手写数字识别模型
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            QInvertedResidual(3, 16, 1, 2), 
            QInvertedResidual(16, 32, 1, 2), 
            QInvertedResidual(32, 64, 2, 2), 
            QInvertedResidual(64, 128, 1, 2), 
            QInvertedResidual(128, 128, 2, 2),
            QInvertedResidual(128, 256, 1, 2)
        )
        # 量化函数
        self.quant = QuantStub()
        # 反量化函数
        self.dequant = DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        x = self.layers(x)
        x = self.dequant(x)
        return x
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == QInvertedResidual:
                m.fuse_model()
# 量化感知训练
model_fp32 = Model() # 32bit浮点
model_fp32.eval()  # 调成测试模型才能融合
# 设置模型后端
#model_fp32.qconfig = torch.quantization.QConfig(
#    activation=torch.quantization.default_observer,
#    weight=torch.quantization.default_per_channel_weight_observer)
model_fp32.qconfig = torch.quantization.QConfig(
    activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), 
    weight=torch.quantization.default_observer.with_args(dtype=torch.qint8))
# 融合模型
model_fp32.fuse_model()
# 融合后调整为训练模式
model_fp32 = model_fp32.train() 
# 准备模型
model_fp32_prepared = torch.quantization.prepare_qat(model_fp32)
# 自行编写训练过程
optim = torch.optim.SGD(model_fp32_prepared.parameters(), 0.1) 

for itr in range(100):
    x = torch.randn([12, 3, 28, 28]) #给定训练数据
    y = model_fp32_prepared(x)
    # TODO:给定训练过程

model_fp32_prepared.eval() # 调整为测试模式
# 8bit模型
model_int8 = torch.quantization.convert(model_fp32_prepared)
import time 
x = torch.randn([50, 3, 64, 64]) 
model = model_fp32 
model.eval() 
t1 = time.perf_counter()
for i in range(10):
    y1 = model(x) 
t2 = time.perf_counter()
model_int8.eval() 
t3 = time.perf_counter()
for i in range(10):
    y1 = model_int8(x) 
t4 = time.perf_counter()
print("Time consumption", f"未量化{t2-t1:.2f}s, 量化后{t4-t3:.2f}s, 加速比{(t2-t1)/(t4-t3):.2f}")

torch.save(model_fp32.state_dict(), "ckpt/fp32.pt") 
torch.save(model_fp32_prepared.state_dict(), "ckpt/fp32_prepared.pt") 
torch.save(model_int8.state_dict(), "ckpt/int8.pt") 