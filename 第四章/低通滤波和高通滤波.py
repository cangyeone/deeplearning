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

# 高通滤波
f1 = np.zeros([N]) 
f1[25:-25] = 1 
y_highpass = np.real(np.fft.ifft(f*f1))
# 低通滤波
f2 = np.ones([N]) 
f2[25:-25] = 0 
y_lowpass = np.real(np.fft.ifft(f*f2))

# 数据坐标制作
time_ = np.linspace(0, S, N) # 1秒数据
freq_ = np.linspace(0, N/S, N) # 最低记录1Hz，最高N/2Hz


gs = grid.GridSpec(3, 6, hspace=0.3, wspace=0.4) 
fig = plt.figure(1, figsize=(12, 9), dpi=100) 
ax = fig.add_subplot(gs[0, :]) 
ax.scatter(time_, y, c="r", label="采样点") 
ax.plot(time_, y, c="k", alpha=0.5, label="波形")
ax.legend(loc="upper right") 
ax.set_xlabel("时间/s")
ax.set_ylabel("振幅")
ax.set_title("a)", x=0.03, y=0.8)
ax = fig.add_subplot(gs[1, 0:2]) 
ax.scatter(freq_[:N//2+1], f_amp[:N//2+1], 
        c="r", label="采样点") 
ax.plot(freq_[:N//2+1], f_amp[:N//2+1], 
        c="k", alpha=0.5, label="原始频谱")
ax.legend(loc="upper right")
ax.set_xlabel("频率/Hz")
ax.set_ylabel("振幅")
ax.set_ylim((0, 300))
ax.set_title("b)", x=0.09, y=0.8)

ax = fig.add_subplot(gs[1, 2:4]) 
ax.scatter(freq_[:N//2+1], (f_amp*f1)[:N//2+1], 
        c="r", label="采样点") 
ax.plot(freq_[:N//2+1], (f_amp*f1)[:N//2+1], 
        c="k", alpha=0.5, label="低频滤除")
ax.legend(loc="upper right")
ax.set_xlabel("频率/Hz")
ax.set_ylim((0, 300))
#ax.set_ylabel("振幅")
ax.set_title("c)", x=0.09, y=0.8)

ax = fig.add_subplot(gs[1, 4:6]) 
ax.scatter(freq_[:N//2+1], (f_amp*f2)[:N//2+1], 
        c="r", label="采样点") 
ax.plot(freq_[:N//2+1], (f_amp*f2)[:N//2+1], 
        c="k", alpha=0.5, label="高频滤除")
ax.legend(loc="upper right")
ax.set_xlabel("频率/Hz")
ax.set_ylim((0, 300))
#ax.set_ylabel("振幅")
ax.set_title("d)", x=0.09, y=0.8)


ax = fig.add_subplot(gs[2, 0:3]) 
#ax.scatter(time_, y, c="r", label="高通滤波后") 
ax.plot(time_, y_highpass, c="k", alpha=0.5, label="高通滤波后")
ax.legend(loc="upper right") 
ax.set_xlabel("时间/s")
ax.set_ylabel("振幅")
ax.set_ylim((-1.1, 1.1))
ax.set_title("e)", x=0.09, y=0.8)

ax = fig.add_subplot(gs[2, 3:6]) 
#ax.scatter(time_, y, c="r", label="低通滤波后") 
ax.plot(time_, y_lowpass, c="k", alpha=0.5, label="低通滤波后")
ax.legend(loc="upper right") 
ax.set_xlabel("时间/s")
ax.set_ylim((-1.1, 1.1))
#ax.set_ylabel("振幅")
ax.set_title("f)", x=0.09, y=0.8)

plt.savefig("导出图像/低通高通滤波.png")
plt.savefig("导出图像/低通高通滤波.svg")