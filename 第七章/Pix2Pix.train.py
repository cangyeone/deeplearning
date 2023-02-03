import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import models.pix2pix as net

from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader 
import numpy as np 
from plotconfig import * 
import pickle 
class ImagetDataset(Dataset):
    def __init__(self):
        self.labels = [] 
        self.images = []
        for i in range(5):
            with open(f"data/cifar-10-batches-py/data_batch_{i+1}", 'rb') as fo:
                data_dict = pickle.load(fo, encoding="bytes")
            d = data_dict[b"labels"]
            x = data_dict[b"data"]
            self.labels.extend(d) 
            self.images.extend(x)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x, d = self.images[idx], self.labels[idx] 
        x = torch.tensor(x, dtype=torch.float32) / 255 
        d = torch.tensor([d], dtype=torch.long) 
        x = x.reshape([3, 32, 32])
        return (x, d) 

def collate_batch(batch):
    """
    定义后处理函数
    """
    xs, ds = [], []
    for x, d in batch:
        xs.append(x) 
        ds.append(d)
    xs = torch.stack(xs, dim=0) 
    ds = torch.cat(ds)
    return xs, ds 

class ConvBNReLU(nn.Module):
    def __init__(self, nin, nout, stride=1):
        super().__init__() 
        self.layers = nn.Sequential(
            nn.Conv2d(nin, nout, 3, stride, padding=1), 
            nn.BatchNorm2d(nout), 
            nn.LeakyReLU()
        )
    def forward(self, x):
        x = self.layers(x)
        return x 

class ConvUpBNReLU(nn.Module):
    # 卷积+上采样层
    def __init__(self, nin, nout, stride=1):
        super().__init__()
        # 使用插值进行上采样
        self.layers = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=stride), 
            nn.Conv2d(nin, nout, 3, stride=1, padding=1), 
            nn.BatchNorm2d(nout), 
            nn.LeakyReLU()
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 
class ImageGeneration(nn.Module):
    def __init__(self, nclass=10):
        super().__init__()
        # 将类别转换为向量
        self.emb = nn.Embedding(nclass, 64)
        self.noize = nn.Linear(100, 64) 
        self.inputs = nn.Linear(128, 64*4*4)
        # 构建生成模型：多层上采样+卷积
        self.layers = nn.Sequential(
            ConvBNReLU(64, 64, 1), 
            ConvUpBNReLU(64, 32, 2), 
            ConvBNReLU(32, 32, 1), 
            ConvUpBNReLU(32, 16, 2), 
            ConvBNReLU(16, 16, 1), 
            ConvUpBNReLU(16, 8, 2), 
            nn.Conv2d(8, 3, 1, 1), 
            nn.Sigmoid() # 输出[0,1]区间
        )
    def forward(self, d, z):
        # 把类别信息进行编码
        h1 = self.emb(d) 
        h2 = self.noize(z) 
        h = torch.cat([h1, h2], dim=1) 
        h = self.inputs(h)
        h = h.reshape([-1, 64, 4, 4])
        x = self.layers(h)
        return x 

class ImageClassify(nn.Module):
    def __init__(self):
        super().__init__() 
        # DNN输出长度为2的表示向量
        self.dnn = nn.Sequential(
            ConvBNReLU( 3, 16, 2), 
            ConvBNReLU(16, 16, 1), 
            ConvBNReLU(16, 32, 2), 
            ConvBNReLU(32, 32, 1), 
            ConvBNReLU(32, 64, 2), 
            ConvBNReLU(64, 64, 1),
            nn.Flatten(), 
            nn.Linear(4*4*64, 128), 
        ) 
        # 在长度为2的表示向量基础上进行分类
        self.classify = nn.Linear(128, 10)
        self.ganout = nn.Linear(128, 1)
    def forward(self, x):
        h = self.dnn(x) # 提取特征向量
        y1 = self.classify(h) 
        y2 = self.ganout(h)
        return y1, y2 

def main():
    device = torch.device("cpu")
    dataset = ImagetDataset() 
    dataloader = DataLoader(dataset, 100, collate_fn=collate_batch)
    # get models
    netG = ImageGeneration(3, 3, 64)
    netD = ImageClassify(6, 64)
    netG.train()
    netD.train()
    criterionBCE = nn.BCELoss()
    criterionCAE = nn.L1Loss()
    

    # get optimizer
    optimizerD = torch.optim.Adam(netD.parameters(), lr = 1e-4, betas = (0.5, 0.999), weight_decay=0.0)
    optimizerG = torch.optim.Adam(netG.parameters(), lr = 1e-4, betas = (0.5, 0.999), weight_decay=0.0)

    # NOTE training loop
    ganIterations = 0
    for target, input in dataloader:
        # max_D first
        for p in netD.parameters(): 
            p.requires_grad = True 
        netD.zero_grad()

        output = netD(torch.cat([target, input], 1)) # conditional
        errD_real = criterionBCE(output, label_d)
        errD_real.backward()
        x_hat = netG(input)
        fake = x_hat.detach()
        output = netD(torch.cat([fake, input], 1)) # conditional
        errD_fake = criterionBCE(output, label_d)
        errD_fake.backward() 
        optimizerD.step() # update parameters

        # prevent computing gradients of weights in Discriminator
        for p in netD.parameters(): 
            p.requires_grad = False
        netG.zero_grad() # start to update G

        # compute L_L1 (eq.(4) in the paper
        L_img_ = criterionCAE(x_hat, target)
        L_img = 0.1 * L_img_
        L_img.backward(retain_variables=True)

        # compute L_cGAN (eq.(2) in the paper
        output = netD(torch.cat([x_hat, input], 1))
        errG_ = criterionBCE(output, label_d)
        errG = 1 * errG_ 
        errG.backward()
        optimizerG.step()
    