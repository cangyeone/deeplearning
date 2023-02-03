import torch 

import matplotlib.image as image 
import numpy as np 
from plotconfig import * 

from torchvision.models.detection import fasterrcnn_resnet50_fpn
rcnn = fasterrcnn_resnet50_fpn(pretrained=True) 
rcnn.eval()

img = image.imread("data/coco2014/images/COCO_train2014_000000549691.jpg")
img = img.astype(np.float32)/255

x = torch.tensor(img).permute(2, 0, 1)
with torch.no_grad():
    y, d = rcnn([x]) 
    d = d[0][0].sigmoid().cpu().numpy()
    y = y[0]
    boxes = y['boxes'].cpu().numpy()
    label = y["labels"].cpu().numpy() 
    score = y["scores"].cpu().numpy()
    

gs = grid.GridSpec(4, 2) 
fig = plt.figure(1, figsize=(6, 10), dpi=100) 
ax = fig.add_subplot(gs[0, 0]) 
ax.imshow(img) 
ax.set_xticks(()) 
ax.set_yticks(()) 
ax.set_title("(a) 原始图像") 
ax = fig.add_subplot(gs[0, 1]) 
p = d[0, :, :]
print(np.max(p), np.min(p))
p[0, 0] = 1
p[1, 1] = 0 
ax.imshow(p, cmap="Greys") 
ax.set_xticks(()) 
ax.set_yticks(()) 
ax.set_title("(b) 第一个Anchor") 
ax = fig.add_subplot(gs[1, 0]) 
p = d[1, :, :]
p[0, 0] = 1
p[1, 1] = 0 
ax.imshow(p, cmap="Greys") 
ax.set_xticks(()) 
ax.set_yticks(()) 
ax.set_title("(c) 第二个Anchor") 
ax = fig.add_subplot(gs[1, 1]) 
p = d[2, :, :]
p[0, 0] = 1
p[1, 1] = 0 
ax.imshow(p, cmap="Greys") 
ax.set_xticks(()) 
ax.set_yticks(()) 
ax.set_title("(d) 第三个Anchor") 

ax = fig.add_subplot(gs[2:, :]) 
ax.imshow(img)
for (x1, y1, x2, y2), d, s in zip(boxes, label, score):
    if s < 0.5:continue 
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], c="r")
    print(s)
ax.set_xticks(()) 
ax.set_yticks(()) 
ax.set_title("(f) 检测结果") 
plt.savefig("导出图像/rpn结果.png")
plt.savefig("导出图像/rpn结果.svg")
plt.show()

