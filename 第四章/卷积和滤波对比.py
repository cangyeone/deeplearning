from tkinter import X
from plotconfig import * 
import numpy as np 
N = 500 # 采样点数量
S = 1 # 数据秒数
t = np.linspace(0, np.pi * 2 * 5, N) 
x1 = np.sin(t)    # 震动较慢为低频，每秒5 个周期为5 Hz
x2 = np.sin(t*10) # 震动较快为高频，每秒50个周期为50Hz
# 制作数据，包含50Hz和5Hz波形，振幅不同
x = x1 * 1 + x2 * 0.1  
# 频谱分析可以分析出频率特征

def conv(x, g):
    # 卷积计算：
    lenx = len(x) 
    leng = len(g) 
    # 对时域滤波器，或者卷积核心需要翻转
    rg = g[::-1]
    y = np.zeros([lenx])
    # 只计算有值位置
    for i in range(lenx-leng):
        y[i] = np.sum(x[i:i+leng]*rg)
    return y 
g = np.ones([10]) / 10

# 时间域滤波
y_time_domain = conv(x, g) 

# 频率域滤波
g_pad = np.pad(g, (0, len(x)-len(g))) # 补零为相同长度
G = np.fft.fft(g_pad) 
X = np.fft.fft(x)
Y = X * G # 频率域直接相乘
y_freq_domain = np.real(np.fft.ifft(Y))


# 数据坐标制作
time_ = np.linspace(0, S, N) # 1秒数据
freq_ = np.linspace(0, N/S, N) # 最低记录1Hz，最高N/2Hz


gs = grid.GridSpec(3, 6, hspace=0.3, wspace=0.4) 
fig = plt.figure(1, figsize=(12, 9), dpi=100) 
ax = fig.add_subplot(gs[0, 0:3]) 
ax.scatter(time_, x, c="r", label="采样点") 
ax.plot(time_, x, c="k", alpha=0.5, label="信号波形")
ax.legend(loc="upper right") 
ax.set_xlabel("时间/s")
ax.set_ylabel("振幅")
ax.set_title("a)", x=0.09, y=0.8)

ax = fig.add_subplot(gs[0, 3:6]) 
ax.scatter(time_, g_pad, c="r", label="采样点") 
ax.plot(time_, g_pad, c="k", alpha=0.5, label="滤波器波形")
ax.legend(loc="upper right") 
ax.set_xlabel("时间/s")
ax.set_ylabel("振幅")
ax.set_title("b)", x=0.09, y=0.8)

ax = fig.add_subplot(gs[1, 0:3]) 
ax.plot(freq_[:N//2+1], np.abs(X)[:N//2+1], 
        c="k", alpha=0.5, label="波形频谱")
ax.legend(loc="upper right")
ax.set_xlabel("频率/Hz")
ax.set_ylabel("振幅")
ax.set_ylim((0, 300))
ax.set_title("c)", x=0.09, y=0.8)

ax = fig.add_subplot(gs[1, 3:6]) 
ax.plot(freq_[:N//2+1], np.abs(G)[:N//2+1], 
        c="k", alpha=0.5, label="滤波器频谱")
ax.legend(loc="upper right")
ax.set_xlabel("频率/Hz")
#ax.set_ylim((0, 300))
#ax.set_ylabel("振幅")
ax.set_title("d)", x=0.09, y=0.8)

ax = fig.add_subplot(gs[2, 0:3]) 
#ax.scatter(time_, y, c="r", label="高通滤波后") 
ax.plot(time_, y_time_domain, c="k", alpha=0.5, label="时间域滤波")
ax.legend(loc="upper right") 
ax.set_xlabel("时间/s")
ax.set_ylabel("振幅")
ax.set_ylim((-1.1, 1.1))
ax.set_title("e)", x=0.09, y=0.8)

ax = fig.add_subplot(gs[2, 3:6]) 
#ax.scatter(time_, y, c="r", label="低通滤波后") 
ax.plot(time_, y_freq_domain, c="k", alpha=0.5, label="频率域滤波")
ax.legend(loc="upper right") 
ax.set_xlabel("时间/s")
ax.set_ylim((-1.1, 1.1))
#ax.set_ylabel("振幅")
ax.set_title("f)", x=0.09, y=0.8)

plt.savefig("导出图像/时间频率域滤波.svg")
plt.savefig("导出图像/时间频率域滤波.png")