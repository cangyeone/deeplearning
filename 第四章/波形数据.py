from plotconfig import * 
import numpy as np 
N = 500 # 采样点数量
S = 1 # 数据秒数
t = np.linspace(0, np.pi * 2 * 5, N) 
y1 = np.sin(t)    # 震动较慢为低频，每秒5 个周期为5 Hz
y2 = np.sin(t*10) # 震动较快为高频，每秒50个周期为50Hz
# 制作数据，包含50Hz和5Hz波形，振幅不同
y = y1 * 1 + y2 * 0.1  
# 频谱分析可以分析出频率特征
f = np.fft.fft(y)  # 制作频谱
f_amp = np.abs(f)  # 频谱为复数，画图为振幅值

# 数据坐标制作
time_ = np.linspace(0, S, N) # 1秒数据
freq_ = np.linspace(0, N/S, N) # 最低记录1Hz，最高N/2Hz


gs = grid.GridSpec(2, 1) 
fig = plt.figure(1, figsize=(12, 9), dpi=100) 
ax = fig.add_subplot(gs[0]) 
ax.scatter(time_, y, c="r", label="采样点") 
ax.plot(time_, y, c="k", alpha=0.5, label="波形")
ax.legend(loc="upper right") 
ax.set_title("a)", x=0.0, y=1.0, va="bottom", ha="left")
ax.set_xlabel("时间/s")
ax.set_ylabel("振幅")

ax = fig.add_subplot(gs[1]) 
ax.scatter(freq_[:N//2+1], f_amp[:N//2+1], 
        c="r", label="采样点") 
ax.plot(freq_[:N//2+1], f_amp[:N//2+1], 
        c="k", alpha=0.5, label="频谱")
ax.legend(loc="upper right")
ax.set_xlabel("频率/Hz")
ax.set_ylabel("振幅")
ax.set_title("b)", x=0.0, y=1.0, va="bottom", ha="left")
plt.savefig("导出图像/波形数据.png")
plt.savefig("导出图像/波形数据.svg")