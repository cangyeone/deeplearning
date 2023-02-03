from plotconfig import * 
import scipy.io.wavfile as wav
import numpy as np 
import scipy.signal 

sr, wave = wav.read("data/data_thchs30/A2_0.wav")
text = "绿是阳春烟景大块文章的底色四月的林峦更是绿得鲜活秀媚诗意盎然"
fig = plt.figure(1, figsize=(12, 12), dpi=100) 
gs = grid.GridSpec(2, 1) 
ax = fig.add_subplot(gs[0, 0]) 
t = np.arange(len(wave)) / sr 
ax.plot(t, wave, c="k", lw=0.5, alpha=0.5, label="波形") 
ax.grid(True) 
ax.set_title(f"标签：{text}\n采样率：{sr}Hz\n时间：{np.max(t):.2}s", x=0.01, y=0.98, va="top", ha="left")
ax.set_ylim((-30000, 30000))
ax.set_ylabel("振幅")
ax.set_xlabel("时间")
ax.set_xlim((0, t.max()))
ax.legend(loc="lower right")
ax.set_title("a)", x=0, y=1, va="bottom", ha="left")

ax = fig.add_subplot(gs[1, 0]) 
f, t, z = scipy.signal.stft(wave, sr, nperseg=512, noverlap=256) 
x = np.log(np.abs(z)+1) 
print(t.shape, f.shape, z.shape, wave.shape)
ax.pcolormesh(t, f, x, cmap="Greys") 
ax.set_ylabel("频率")
ax.set_xlabel("时间")
ax.set_title("b)", x=0, y=1, va="bottom", ha="left")


plt.savefig("导出图像/波形示意.STFT.jpg")
plt.savefig("导出图像/波形示意.STFT.svg")
import torch 
x = wave 
# scipy完成短时傅里叶变换
# f为频率坐标，t为时间坐标，z为结果
f, t, z = scipy.signal.stft(x, 16000, nperseg=512, noverlap=256)
z = np.log(np.abs(z)+1) 
# torch的API完成变换
x_torch = torch.tensor(x, dtype=torch.float32) 
# 变换后数据格式为[C, T, 2]，分别代表复数实部和虚部
z_torch = torch.stft(
    x_torch, n_fft=512, hop_length=256, return_complex=False)
# 计算复数模，并取log
z_torch = z_torch.square().sum(dim=2).sqrt().log()
print(z_torch.shape)

