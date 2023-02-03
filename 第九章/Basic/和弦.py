import numpy as np 
import wave 
import math 
import scipy.signal as signal 
import matplotlib.pyplot as plt 
def gen_sin(amp, f, fs, tau):
    nT = np.linspace(0, tau, round(tau/(1.0/fs)))
    signal = amp*np.cos(2*np.pi*f*nT)
    return signal
 
#model the harmonic feature in frequency domain
Amp=[1,0.340,0.102,0.085,0.070,0.065,0.028,0.085,0.011,0.030,0.010,0.014,0.012,0.013,0.004]
numharmonic=len(Amp)#谐波个数
 
pianomusic=np.zeros([40000])
startpoint=0

#model the piano note attenuation feature in the time domain
attenuation = np.zeros([8000])

#the attack stage
attenuation[:200] = np.arange(0, 200)*0.005
#the attenuate stage
attenuation[200:800] = 1-np.arange(0, 600)*0.001
#the maintain stage
attenuation[800:4000] = 0.4 - np.arange(0, 3200) * 0.000078
attenuation[4000:8000] = 0.15 - np.arange(0, 4000) * 0.0000078

#compose each note in each time quantum
nomalizedbasicfreq=[261.63,261.63,261.63,261.63,293.665,293.665,293.665,293.665,329.628,329.628,329.628,329.628,349.228,349.228,349.228,349.228,391.995,391.995,391.995,391.995,440,440,440,440,493.883,493.883,493.883,493.883,523.251,523.251,523.251,523.251,587.33,587.33,587.33,587.33,659.255,659.255,659.255,659.255]
ampli=[(math.pow(2,2*8-1)-1) for i in range(0,40)]
#40个/4=10
notestime=[4,4,4,4,4,4,4,4,4,4]#10个
windowsize=1000
for w in range(0,len(notestime)):
    pianonote = np.zeros(windowsize*notestime[w]) #get the length according to the time of the note
    for i in range(0, numharmonic): #get the note by add each harmonic by the amplitude comparatively with the basic frequency
        pianonote = pianonote + gen_sin(ampli[startpoint] /50* Amp[i], nomalizedbasicfreq[startpoint] * (i + 1), 8000, 0.125*notestime[w])
    #attenuate the note with the time domain feature
    for k in range(0,windowsize*notestime[w]):#k:0---4000
        pianomusic[startpoint*windowsize+k]=pianonote[k]*attenuation[k]
        #0--4000 startpoint=0
        #4000-8000   =4
        #8000-12000  =8
    startpoint=startpoint+notestime[w] #record the start point of the next note
    #startpoint变化规律：0,4,8,12，...32，36
 
plt.plot(pianomusic)
plt.show() 
f, t, Z = signal.stft(pianomusic, 8000, nperseg=512) 
plt.pcolormesh(t, f, np.abs(Z)) 
plt.show()
f = wave.open("pianomusic.wav", "wb")
f.setnchannels(1)
f.setsampwidth(2)
f.setframerate(8000)
f.writeframes((pianomusic*3).astype(np.int16).tostring())
f.close()