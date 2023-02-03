#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
=====================
MFCC
=====================
MFCC特征处理过程
"""
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
import scipy.fftpack as fft
import matplotlib.style as style
import matplotlib
style.use("ggplot")
matplotlib.rcParams['font.sans-serif'] = ['SimHei']   
matplotlib.rcParams['font.family']='sans-serif' 

n_mel_feature = 13
rate, sig = wav.read("data/data_thchs30/A2_0.wav")
label = open("data/data_thchs30/A2_0.wav.trn", "r", encoding="utf-8").read()
plt.figure(1)
xx = np.linspace(0, len(sig)/rate, len(sig))
plt.plot(xx, sig)
plt.text(0,10000,"SampleRate:%d\nLenth:%d\nTimes:%.4fs\n"%(rate, 
         len(sig), 
         len(sig)/rate))
plt.xlabel("Time(s)")
plt.show()

# 以类差分的方式增强高频信号
signal_en = sig[1:]-0.97*sig[:-1]
plt.figure(2)
plt.subplot(211)
xx_en = np.linspace(0, len(signal_en)/rate, len(signal_en))
plt.plot(xx, sig, alpha=0.3, color="#990000", label="原始数据")
plt.plot(xx_en, signal_en, alpha=0.3, color="#009900", label="高频增强")
plt.legend()
plt.subplot(212)
plt.plot(xx_en[160000:160000+160], signal_en[160000:160000+160], alpha=0.3, color="#009900", label="帧")
plt.legend()
plt.plot()
plt.show()

# 得到Frame
plt.figure(3)
sig_en_len = len(signal_en)
frame = np.zeros([sig_en_len//80, 200])
for idx, itr in enumerate(frame):
    if idx == len(frame)-1:
        t_d = signal_en[idx*80:]
        frame[idx, :len(t_d)] = t_d
        break
    #frame[idx, :] = signal_en[idx*80:idx*80+201]
plt.matshow(np.transpose(frame))
plt.show()

# 得到Frame,进行FFT变换，上面过程可以称之为STFT
frame_f = np.fft.fft(frame, axis=1)
# 能量
frame_E = np.square(np.abs(frame_f))/400
plt.matshow(np.transpose(np.log(frame_E)))
plt.show()

frame_F = np.sum(frame_E, axis=1)

fft_x = np.linspace(0, rate, rate)
mel_x = 1125 * np.log(1 + fft_x/700)
plt.figure(4)
plt.subplot(111)
plt.plot(fft_x, mel_x)
plt.show()

mel_f = np.linspace(0, mel_x[-1], n_mel_feature+2)
mel_to_fq = 700*(np.exp(mel_f/1125)-1)
# 绘制映射过程
plt.figure(5)
plt.subplot(111)
plt.plot(fft_x, mel_x)
for itx, ity in zip(mel_to_fq, mel_f):
    plt.plot([0, itx, itx], [ity, ity, 0], color="#006600")
plt.xlabel("傅里叶频率")
plt.ylabel("梅尔频率")
plt.show()


fft_xi = np.linspace(0, rate, 200)
mel_filter_for_plot = []
mel_f_idx = []
mel_f_idx.append(0)
for itr in range(n_mel_feature):
    #tcz = np.zeros()
    c_idx = np.where(np.abs(mel_to_fq[itr+1]-fft_xi)==
             np.min(np.abs(mel_to_fq[itr+1]-fft_xi)))[0][0]
    mel_f_idx.append(c_idx)
mel_f_idx.append(199)

# 绘制带通滤波器
plt.figure(6)
plt.subplot(111)
print(mel_f_idx)
filters = []
for itr in range(n_mel_feature):
    cidx = np.zeros(200)
    cidx[mel_f_idx[itr]:mel_f_idx[itr+1]+1] = np.linspace(0, 1, mel_f_idx[itr+1]-mel_f_idx[itr] + 1)
    cidx[mel_f_idx[itr+1]:mel_f_idx[itr+2]+1] = np.linspace(1, 0, mel_f_idx[itr+2]-mel_f_idx[itr+1] + 1)
    filters.append(cidx)
    plt.plot(fft_xi, cidx)
plt.show()

# 积分获取能量
energy = []
for itr in frame_E:
    itfram = []
    for ft in filters:
        itfram.append(np.sum(ft*itr))
    energy.append(itfram)

eng_log = np.log(energy)
# 离散余弦变换
mfcc = fft.dct(eng_log, axis=1)
plt.figure(7)
plt.matshow(np.log(np.transpose(np.abs(mfcc[:50, :]))+1))
plt.show()
        



