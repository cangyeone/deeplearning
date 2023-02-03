
import torch 
import torch.nn as nn 
from models.networks import ResnetGenerator 
import cv2 
from plotconfig import * 
import numpy as np 
import tqdm 

genA = ResnetGenerator(3, 3, norm_layer=nn.modules.instancenorm.InstanceNorm2d, ngf=64, n_blocks=9) 
genA.eval() 
genA.load_state_dict(torch.load("ckpt/200_net_G_A.pth"))

fig = plt.figure(1, figsize=(12, 6), dpi=100) 
gs = grid.GridSpec(2, 4)
for i in range(1):
    for j in range(4):
        ax = fig.add_subplot(gs[i, j]) 
        img = plt.imread(f"data/scen/{i*2+j}.jpg")
        ax.imshow(img)
        ax.set_xticks(()) 
        ax.set_yticks(())
        img = (img.astype(np.float32)[:, :, ::1]) / 255. 
        x = torch.tensor(img, dtype=torch.float32)
        x = x.unsqueeze(0) 
        x = x.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            y = genA(x)
            y = y.permute(0, 2, 3, 1)
            y = (y.cpu().numpy()[:, :, ::1] + 1)/2 * 255
            y = y[0].astype(np.uint8)
        ax = fig.add_subplot(gs[i+1, j]) 
        ax.imshow(y) 
        ax.set_xticks(()) 
        ax.set_yticks(())
plt.savefig("导出图像/CycleGAN结果.jpg")
plt.savefig("导出图像/CycleGAN结果.svg")

