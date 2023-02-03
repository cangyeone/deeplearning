import torch 
import torch.nn as nn 

x = torch.zeros([5, 128, 28, 28]) 
cnn = nn.Conv2d(128, 256, 3, groups=2) 
for var in cnn.parameters():
    print(var.shape)

cnn = nn.Conv1d(128, 256, 3, groups=2) 
for var in cnn.parameters():
    print(var.shape)