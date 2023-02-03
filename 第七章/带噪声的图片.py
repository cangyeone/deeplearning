from plotconfig import * 
import numpy as np 
import cv2 

img1 = cv2.imread("data/coco2014/images/COCO_train2014_000000534168.jpg")

img_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV) 
img_hsv = img_hsv.astype(np.float32)/255 
img_hsv[:, :, 2] *= 0.5 
img_hsv = (img_hsv * 255).astype(np.uint8) 


img2 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
img2 = img2.astype(np.float32) / 255 
img2 += np.random.uniform(0, 0.2, img2.shape) 
img2 = np.clip(img2, 0, 1) 
img2 = (img2*255).astype(np.uint8)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) 

fig = plt.figure(1, figsize=(12, 6), dpi=100) 
gs = grid.GridSpec(1, 2) 
ax = fig.add_subplot(gs[0, 0]) 
ax.imshow(img1) 
ax.set_xticks(()) 
ax.set_yticks(())
ax.set_title("原始图像")
ax = fig.add_subplot(gs[0, 1]) 
ax.imshow(img2)
ax.set_xticks(()) 
ax.set_yticks(())
ax.set_title("降低亮度+添加噪声")
plt.savefig("导出图像/图像滤波.jpg")
plt.savefig("导出图像/图像滤波.svg")
plt.show()



