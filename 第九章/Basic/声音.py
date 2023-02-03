"""
信号滤波演示
"""
import wave 
import numpy as np 
import scipy.signal as signal 
import matplotlib.pyplot as plt 
# 绘制一段三角函数
x = np.linspace(0, 2*np.pi, 100000) # 60000 称为采样点数量
#500的话，音频是5秒，每秒钟有100个周期，此时称音频频率为100Hz
freq = "noise" 
y = np.sin(x * 6000) + np.random.normal(0, 0.3, [100000]) # 高斯白噪声
plt.plot(x, y)
plt.show()


f, t, Z = signal.stft(y, 20000, nperseg=1024, noverlap=1024-256) 
#import librosa 
#M = librosa.feature.mfcc(y, sr=20000)
#print("特征shape", Z.shape, M.shape)
plt.pcolormesh(t, f, np.abs(Z)) 
plt.show()

y *= 20000
y = y.astype(np.int16)
files = wave.open(f"test-{freq}.wav", "wb") 
files.setnchannels(1) # 通道数1，立体声通道数2 
files.setsampwidth(2) # 比特率16，CD音质24比特
files.setframerate(20000) # 采样率，每秒钟有多少个采样点，CD音质44100（Hz）44.1KHz
files.writeframes(y.tostring()) 
files.close()