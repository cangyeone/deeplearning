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

fig = plt.figure(1, figsize=(9, 9), dpi=100)
gs = grid.GridSpec(1, 4, figure=fig)
gs1 = grid.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0, :3])
gs2 = grid.GridSpecFromSubplotSpec(3, 1, wspace=0.2, hspace=0.2, subplot_spec=gs[0, 3])
ax = fig.add_subplot(gs1[0, 0])
ax.imshow(img)
H, W, C = img.shape
ax.set_title("a)", x=0, y=1, va="bottom", ha="left")
set_axis(ax)

names = ["b)", "c)", "d)"]
for i in range(3):
    ax = fig.add_subplot(gs2[i])
    img1 = img.copy()
    for k in range(3):
        if i == k:continue 
        img1[:, :, k] = 0
    ax.imshow(img1)
    ax.set_title(f"{names[i]}", x=0, y=1, va="bottom", ha="left")
    set_axis(ax)

plt.savefig("导出图像/图像通道.png")
plt.savefig("导出图像/图像通道.svg")
plt.show()