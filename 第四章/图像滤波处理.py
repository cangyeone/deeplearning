from plotconfig import * 
import numpy as np 
import cv2 
#彩色图像
img = cv2.imread("data/img.jpg")
img = img.astype(np.float32)/255 #0-1区间浮点数

img_gray = np.mean(img, axis=2)[::4, ::4] # [H, W, C]格式


def conv2d(x, g):
    # 二维卷积
    h, w = x.shape 
    k1, k2 = g.shape 
    # 卷积核心翻转
    g = g[::-1, ::-1]
    y = np.zeros([h, w]) 
    for i in range(h-k1):
        for j in range(w-k2):
            y[i, j] = \
                np.sum(x[i:i+k1, j:j+k2]*g)
    return y 
# 定义滤波器
g = np.ones([4, 4])
g[:2, :] = -1 
# 时间域滤波
y1 = conv2d(img_gray, g)

# 频率域滤波
X = np.fft.fft2(img_gray) # 二维傅里叶变换
h, w = img_gray.shape 
k1, k2 = g.shape
# 需要补0，使得二者维度相同
g_pad = np.pad(g, ((0, h-k1), (0, w-k2)))
G = np.fft.fft2(g_pad) # 二维傅里叶变换
Y = X * G # 频谱乘法等价于时间域卷积 

y = np.fft.ifft2(Y) 
y2 = np.real(y) 

y1[0, 0] = -1 
y1[0, 1] = 1 
y2[0, 0] = -1 
y2[0, 1] = 1 
gs = grid.GridSpec(1, 3, hspace=0.3, wspace=0.2) 
fig = plt.figure(1, figsize=(12, 6), dpi=100) 
ax = fig.add_subplot(gs[0]) 
ax.matshow(img_gray, cmap="gray") 
ax.set_xticks(()) 
ax.set_yticks(()) 
ax.set_title("a)", x=0, y=1, va="bottom", ha="left")

ax = fig.add_subplot(gs[1]) 
ax.matshow(y1, cmap="gray") 
ax.set_xticks(()) 
ax.set_yticks(()) 
ax.set_title("b)", x=0, y=1, va="bottom", ha="left")

ax = fig.add_subplot(gs[2]) 
ax.matshow(y2, cmap="gray") 
ax.set_xticks(()) 
ax.set_yticks(()) 
ax.set_title("c)", x=0, y=1, va="bottom", ha="left")

plt.savefig("导出图像/图像滤波处理.png")
plt.savefig("导出图像/图像滤波处理.svg")


