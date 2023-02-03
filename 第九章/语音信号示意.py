from plotconfig import * 
import scipy.io.wavfile as wav
import numpy as np 

sr, wave = wav.read("data/data_thchs30/A2_0.wav")
text = "绿是阳春烟景大块文章的底色四月的林峦更是绿得鲜活秀媚诗意盎然"
fig = plt.figure(1, figsize=(12, 6), dpi=100) 
gs = grid.GridSpec(1, 1) 
ax = fig.add_subplot(gs[0, 0]) 
t = np.arange(len(wave)) / sr 
ax.plot(t, wave, c="k", lw=0.5, alpha=0.5, label="波形") 
ax.grid(True) 
ax.set_title(f"标签：{text}\n采样率：{sr}Hz\n时间：{np.max(t):.2}s", x=0.01, y=0.98, va="top", ha="left")
ax.set_ylim((-30000, 30000))
ax.set_ylabel("振幅")
ax.set_xlabel("时间")
ax.legend(loc="lower right")
plt.savefig("导出图像/波形示意.jpg")
plt.savefig("导出图像/波形示意.svg")
