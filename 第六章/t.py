import numpy as np
from disba import PhaseDispersion, PhaseSensitivity 

# Velocity model
# thickness, Vp, Vs, density
# km, km/s, km/s, g/cm3
velocity_model = np.array([
   [10.0, 7.00, 3.50, 2.00],
   [10.0, 6.80, 3.40, 2.00],
   [10.0, 7.00, 3.50, 2.00],
   [10.0, 7.60, 3.80, 2.00],
   [10.0, 8.40, 4.20, 2.00],
   [10.0, 9.00, 4.50, 2.00],
   [10.0, 9.40, 4.70, 2.00],
   [10.0, 9.60, 4.80, 2.00],
   [10.0, 9.50, 4.75, 2.00],
])

# Periods must be sorted starting with low periods
t = np.logspace(0.0, 3.0, 100)

# Compute the 3 first Rayleigh- and Love- wave modal dispersion curves
# Fundamental mode corresponds to mode 0
ps = PhaseSensitivity(*velocity_model.T)
parameters = ["thickness", "velocity_p", "velocity_s", "density"]
a = ps(20, mode=0, wave="rayleigh", parameter="thickness")
depth = a.depth 
kernel = a.kernel#dC/dh
print(depth, kernel)
old_ckpt = # 原始模型的参数
new_ckpt = {}
for k, v in model.named_parameters():
   # state_dict()相当于仅保留了数值而非可训练参数
   # 应当使用named_parameters()
   if "output" in k:
       # 名字中有output的参数可训练
       v.requires_grad_(True)
       new_ckpt[k] = v.data 
   else:
       # 卷积层不需要重新训练
       v.requires_grad_(False)
       # 加载之原始模型中的参数
       new_ckpt[k] = old_ckpt[k]
for k, v in model.named_buffers():
   # buffer即不可训练的参数
   # 比如BN层中的均值、方差
   new_ckpt[k] = old_ckpt[k]
# 模型加载可训练参数
model.load_state_dict(new_ckpt)
# 定义优化器，仅传入可求导的部分
optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))