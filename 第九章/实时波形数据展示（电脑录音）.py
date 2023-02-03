import numpy as np 
import pyaudio 
import cv2 
from sklearn.externals import joblib 


# 加载机器学习模型
#pca = joblib.load("pca_model") 
#svm = joblib.load("svm_model")


def get_frame(data):
    """
    展示1s波形
    """
    return
    

# 实时读取音频数据
CHUNK = 1600 
FORMAT = pyaudio.paInt16
CHANNELS = 1 
RATE = 16000 
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
# 读取音频数据并进行展示
datas = []
while True:
    data = stream.read(CHUNK) # 从话筒获取CHUNK个点
    datas.append(data)
    if len(datas)<10:continue
    data = np.frombuffer(b''.join(datas[-10:]),dtype = np.short)# 将最近获得的10个CHUNK连接起来作为数据
    data = np.array(data, np.float)#二进制数据转换为浮点类型数据
    maxd = (np.max(data))
    data /= (np.max(data))
    img = get_frame(data)
    cv2.imshow("img", img)
    cv2.waitKey(100)
    if len(datas)>20:
        datas = datas[-20:]   
        