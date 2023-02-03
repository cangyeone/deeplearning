import cv2 
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid 
import numpy as np 

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['font.size'] = "16"  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号



def set_axis(ax):
    ax.set_xticks(())
    ax.set_yticks(())

img = cv2.imread("data/img2.jpg") # 数据读取
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 通道顺序调整
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # 变为HSV
# 调整色相
img_hsv0 = img_hsv.copy()
# 取值范围0-180 
h = img_hsv0[:, :, 0].copy().astype(np.float32)/179 
h += 120/360 #色相整体调整120度 
h = h % 1 # 取值范围调整到0-1，numpy中%可以用于浮点数 
h *= 180 
h = h.astype(np.uint8)
img_hsv0[:, :, 0] = h # 饱和度变为原来的一半

# 调整饱和度
img_hsv1 = img_hsv.copy()
s = img_hsv1[:, :, 1].copy().astype(np.float32)/255 
s *= 0.5 
s = np.clip(s, 0, 1) * 255 # 防止出现数值溢出问题
s = s.astype(np.uint8)
img_hsv1[:, :, 1] = s # 饱和度变为原来的一半
# 调整明度
img_hsv2 = img_hsv.copy()
v = img_hsv2[:, :, 2].copy().astype(np.float32)/255 
v *= 1.5 # 浮点数据更加容易处理
v = np.clip(v, 0, 1) * 255 # 防止出现数值溢出问题
v = v.astype(np.uint8)
img_hsv2[:, :, 2] = v # 明度变为原来的1.5倍

# 最后再变回RGB色彩进行绘图
img_rgb0 = cv2.cvtColor(img_hsv0, cv2.COLOR_HSV2RGB) 
img_rgb1 = cv2.cvtColor(img_hsv1, cv2.COLOR_HSV2RGB) 
img_rgb2 = cv2.cvtColor(img_hsv2, cv2.COLOR_HSV2RGB) 

fig = plt.figure(1, figsize=(12, 9))
gs = grid.GridSpec(2, 2, figure=fig)
ax = fig.add_subplot(gs[0])
ax.imshow(img)
ax.set_title(rf"a)", x=0, y=1, va="bottom", ha="left")
set_axis(ax)

ax = fig.add_subplot(gs[1])
ax.imshow(img_rgb0)
ax.set_title(rf"b)", x=0, y=1, va="bottom", ha="left")
set_axis(ax)

ax = fig.add_subplot(gs[2])
ax.imshow(img_rgb1)
ax.set_title(rf"c)", x=0, y=1, va="bottom", ha="left")
set_axis(ax)
ax = fig.add_subplot(gs[3])
ax.imshow(img_rgb2)
ax.set_title(rf"d)", x=0, y=1, va="bottom", ha="left")
set_axis(ax)
plt.savefig("导出图像/图像HSV调整.png")
plt.savefig("导出图像/图像HSV调整.svg")
plt.show()