import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss 
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader 
from torch.nn.utils.rnn import pad_sequence
import os 
import numpy as np 
import re 
import cv2 

class ImagetDataset(Dataset):
    def __init__(self, file_dir="data/coco/"):
        file_names = os.listdir(file_dir) 
        paths = [] 
        for fn in file_names:
            if fn.endswith(".jpg") == False:continue 
            paths.append(os.path.join(file_dir, fn))
        self.paths = paths 

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img1 = cv2.imread(self.paths[idx]) 
        if len(img1.shape)<3:
            img1 = np.zeros([256, 256, 3], dtype=np.uint8) 
        h, w, c = img1.shape 
        if c != 3:
            img1 = np.zeros([256, 256, 3], dtype=np.uint8) 
        if h<=256 or w <=256:
            img1 = cv2.resize(img1, (256, 256)) 
        img1 = img1[:256, :256]
        img_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV) 
        img_hsv = img_hsv.astype(np.float32)/255 
        img_hsv[:, :, 2] *= np.random.uniform(0.5, 0.9)
        img_hsv = (img_hsv * 255).astype(np.uint8) 

        img2 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        img2 = img2.astype(np.float32) / 255 
        img2 += np.random.uniform(0, np.random.uniform(0.1, 0.3), img2.shape) 
        img2 = np.clip(img2, 0, 1) 
        #img2 = (img2*255).astype(np.uint8)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  
        img1 = img1.astype(np.float32)/255        
        sample = (torch.Tensor(img2).permute(2, 0, 1).float(), torch.Tensor(img1).permute(2, 0, 1).float())
        return sample

def collate_batch(batch):
    """
    定义后处理函数
    """
    xs, ds = [], []
    for x, d in batch:
        xs.append(x) 
        ds.append(d)
    xs = torch.stack(xs, dim=0) 
    ds = torch.stack(ds, dim=0)
    return xs, ds 

class Conv2d(nn.Module):
    def __init__(self, nin=8, nout=11, ks=3, st=2, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(nin, nout, ks, st, padding=padding), 
            nn.BatchNorm2d(nout), 
            nn.ReLU()
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 

class Conv2dT(nn.Module):
    def __init__(self, nin=8, nout=11, ks=3, st=2, padding=1, output_padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2), 
            Conv2d(nin, nout, ks, 1, padding=padding), 
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = Conv2d(3, 8, 3, 1, padding=1) 
        self.layer0 = Conv2d(8, 8, 3, 1, padding=1) 
        self.layer1 = Conv2d(8, 16, 3, 2, padding=1)
        self.layer2 = Conv2d(16, 32, 3, 2, padding=1)
        self.layer3 = Conv2d(32, 64, 3, 2, padding=1) 
        self.layer4 = Conv2d(64, 128, 3, 2, padding=1) 
        self.layer5 = Conv2dT(128, 64, 3, 2, padding=1, output_padding=1)
        self.layer6 = Conv2dT(128, 32, 3, 2, padding=1, output_padding=1)
        self.layer7 = Conv2dT(64, 16, 3, 2, padding=1, output_padding=1)
        self.layer8 = Conv2dT(32, 8, 3, 2, padding=1, output_padding=1)
        self.layer9 = nn.Conv2d(16, 3, 3, 1, padding=1)
    def forward(self, x):
        x = self.inputs(x)
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4) 
        x6 = self.layer5(x5)
        x6 = torch.cat([x4, x6], dim=1) 
        x7 = self.layer6(x6)
        x7 = torch.cat([x3, x7], dim=1) 
        x8 = self.layer7(x7)
        x8 = torch.cat([x2, x8], dim=1) 
        x9 = self.layer8(x8)
        x9 = torch.cat([x1, x9], dim=1) 
        x10 = self.layer9(x9)
        x10 = x10.sigmoid()
        return x10
import matplotlib.pyplot as plt 
import matplotlib.gridspec as grid 

def main():
    train_dataset = ImagetDataset("data/train2014")     
    train_dataloader = DataLoader(train_dataset, batch_size=40, shuffle=True, collate_fn=collate_batch, num_workers=3)

    gpu = False   #使用GPU 

    model = UNet()
    model.train() 
    if gpu:
        model.cuda() 
    model.load_state_dict(torch.load("ckpt/denoise.pt"))
    optim = torch.optim.Adam(model.parameters(), 1e-3)
    lossfn = nn.MSELoss()
    n_epoch = 100
    count = 0
    for b in range(n_epoch):
        for x, d in train_dataloader:
            if gpu:
                x = x.cuda()
                d = d.cuda()
            #x = x.permute(1, 0) 
            #d = d.permute(1, 0)
            #print(x.shape, d.shape)
            y = model(x) 
            # T, B, C->B, C, T 
            loss = lossfn(y, d) 
            loss.backward() 
            optim.step() 
            optim.zero_grad()
            count += 1
            # 今天 学习 的 是 深度循环神经网络
            if count % 50 ==1:
                x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
                p = y.permute(0, 2, 3, 1).detach().cpu().numpy() 
                d = d.permute(0, 2, 3, 1).detach().cpu().numpy() 
                fig = plt.figure()
                gs = grid.GridSpec(1, 3) 
                ax = fig.add_subplot(gs[0]) 
                ax.imshow(x[0])
                ax = fig.add_subplot(gs[1]) 
                ax.imshow(d[0])
                ax = fig.add_subplot(gs[2]) 
                ax.imshow(p[0])   
                plt.savefig("demo.jpg")
                plt.close()             
                print(b, count, loss.detach().cpu().numpy())
                torch.save(model.state_dict(), "ckpt/denoise.pt")

if __name__ == "__main__":
    main()