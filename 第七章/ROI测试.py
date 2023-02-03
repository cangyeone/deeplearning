from torchvision.ops import roi_pool 
import numpy as np 
import torch 

x = torch.zeros([1, 2, 7, 9])
x[0, 0, :, :] = torch.arange(0, 9, 1) 
x[0, 1, :, :] = torch.arange(0, 7, 1)[..., None]
box = torch.tensor([[0, 0, 9, 7]]).float()
sp = torch.tensor([2, 2]).long()
print(7/2, 9/2)
out = roi_pool(x, boxes=[box], output_size=sp)
print(out[0, 0])
print(out[0, 1])