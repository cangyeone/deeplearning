import cv2 
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid 
import numpy as np 

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['font.size'] = "16"  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

img = cv2.imread("data/img.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

def set_axis(ax):
    ax.set_xticks(())
    ax.set_yticks(())

fig = plt.figure(1, figsize=(14, 9), dpi=100)
gs = grid.GridSpec(1, 2, figure=fig)
gs1 = grid.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0])
gs2 = grid.GridSpecFromSubplotSpec(2, 2, wspace=0.2, hspace=0.2, subplot_spec=gs[1])
ax = fig.add_subplot(gs1[0, 0])
ax.imshow(img)
ax.set_title("a)", x=0.0, y=1.0, va="bottom", ha="left")
set_axis(ax)

w1 = np.ones([32, 32]) / 32 / 32 
w2 = np.ones([16, 16]) 
w2[:8, :] = -1 
w3 = np.ones([16, 16]) 
w3[:, :8] = -1 
w4 = np.ones([16, 16]) 
w4[:8, :8] = -1 
w4[8:, 8:] = -1 
filters = [w1, w2, w3, w4]
names = ["b)","c)","d)","e)"]
for i in range(4):
    ax = fig.add_subplot(gs2[i])
    weight = filters[i]
    img1 = cv2.filter2D(img, -1, weight)[::2, ::2, :]
    ax.imshow(img1)
    ax.set_title(f"{names[i]}",x=0.0,y=1.0, ha="left", va="bottom")
    set_axis(ax)

plt.savefig("导出图像/图像特征.png")
plt.savefig("导出图像/图像特征.svg")
plt.show()