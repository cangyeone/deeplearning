import pickle 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader 
import numpy as np 
from plotconfig import * 

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
        x = torch.tensor(x, dtype=torch.float32) / 128 - 1 
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
            nn.ReLU()
        )
    def forward(self, x):
        x = self.layers(x)
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
            nn.Linear(4*4*64, 2), 
        ) 
        # 在长度为2的表示向量基础上进行分类
        self.classify = nn.Linear(2, 10)
    def forward(self, x):
        h = self.dnn(x) # 提取特征向量
        y = self.classify(h) 
        return y, h 

import torch.nn.functional as F 
import math 
class ArcLoss(nn.Module):
    def __init__(self, nfeature, nclass, m=0.5, s=5):
        """Build an additive angular margin loss object for Keras model."""
        super().__init__()
        # 设定文章中的m,s
        self.register_buffer(
            "m", torch.tensor([m], dtype=torch.float32)
        ) 
        self.register_buffer(
            "s", torch.tensor([s], dtype=torch.float32)
        ) 
        # 可训练参数w 
        self.register_parameter(
            "w", nn.Parameter(torch.randn([nfeature, nclass]))
        )
        self.nclass = nclass 
    def forward(self, h, d):
        mask = F.one_hot(d, self.nclass).float()
        # 对向量进行归一化，使得长度为1
        h_norm = F.normalize(h, dim=1) 
        w_norm = F.normalize(self.w, dim=0)
        # 计算归一化向量乘法，即角度
        coso = h_norm @ w_norm 
        sino = (1-coso**2).sqrt() 
        # 计算cos(theta+m)
        sinm = torch.sin(self.m) 
        cosm = torch.cos(self.m)
        cosom = coso * cosm - sino * sinm 
        print(cosom)
        exp_cosom = torch.exp((cosom * mask).sum(dim=1) * self.s)
        exp_coso = torch.exp(coso * self.s) * (1-mask)
        sum_exp = exp_coso.sum(dim=1) 
        #print(exp_cosom, sum_exp)
        loss = -torch.log(exp_cosom/(exp_cosom+sum_exp)).mean()
        return loss



    
from plotconfig import * 
def main():
    dataset = ImagetDataset() 
    dataloader = DataLoader(dataset, 100, collate_fn=collate_batch)
    model = ImageClassify() 
    model.train() 
    lossfn = nn.CrossEntropyLoss() 
    lossce = ArcLoss(2, 19, 0.1)
    optim = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=0.000) 
    step = 0 
    try:
        model.load_state_dict(torch.load("ckpt/imageclassify.ss"))
    except:
        pass 
    for e in range(1):
        for x, d in dataloader:
            y, h = model(x) 
            loss2 = lossce(h, d)
            loss = loss2 
            loss.backward() 
            optim.step() 
            optim.zero_grad()
            step += 1 
            if step % 50 ==0:
                torch.save(model.state_dict(), "ckpt/imageclassify.arcloss") 
                print(loss)
            break 
        fig = plt.figure(1, figsize=(9, 9)) 
        gs = grid.GridSpec(1, 1) 
        ax = fig.add_subplot(gs[0]) 
        h = h.detach().cpu().numpy() 
        d = d.detach().cpu().numpy() 
        names = ["飞机", "手机", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
        #center = lossce.center
        #center = center.cpu().numpy()
        for i in range(10):
            v = h[d==i]
            #c = center[i]
            ax.scatter(v[:, 0], v[:, 1], marker=f"${i}$", s=200, c="k", alpha=0.5, label=names[i]) 
            t = np.linspace(0, np.pi*2, 1000)
            r = (np.max(v)-np.min(v))/4
            #x = c[0] + np.cos(t) * r 
            #y = c[1] + np.sin(t) * r
            #ax.plot(x, y, c="k", linestyle="--", lw=1.5, alpha=0.6)
            #ax.plot(x, y, c="k", linestyle="--", lw=1.5, alpha=0.6)
        #ax.plot([], [], c="k", linestyle="--", lw=1.5, alpha=0.6, label="不同类别")
        ax.legend(loc="upper right", ncol=2) 
        max = np.max(np.abs(h)) 
        ax.set_xlim(([-max, max]))
        ax.set_ylim(([-max, max]))
        plt.savefig("导出图像/CIFAR分类.ArcLoss.jpg")
        plt.savefig("导出图像/CIFAR分类.ArcLoss.svg")
        plt.close()
        print(e, "绘图完成")



if __name__ == "__main__":
    main()