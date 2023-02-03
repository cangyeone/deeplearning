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

class ImageClassify2(nn.Module):
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
class ImageClassify(nn.Module):
    def __init__(self):
        super().__init__() 
        # DNN输出长度为2的表示向量
        self.dnn = nn.Sequential(
            ConvBNReLU( 3, 16, 2), 
            #ConvBNReLU(16, 16, 1), 
            ConvBNReLU(16, 32, 2), 
            #ConvBNReLU(32, 32, 1), 
            ConvBNReLU(32, 64, 2), 
            #ConvBNReLU(64, 64, 1),
            ConvBNReLU(64, 128, 2), 
            #ConvBNReLU(128, 128, 1),
            ConvBNReLU(128, 256, 2), 
            #ConvBNReLU(256, 256, 1),
            nn.Flatten(), 
            nn.Linear(1*1*256, 2), 
        ) 
        # 在长度为2的表示向量基础上进行分类
        self.classify = nn.Linear(2, 10)
    def forward(self, x):
        h = self.dnn(x) # 提取特征向量
        y = self.classify(h) 
        return y, h 
class CenterLoss(nn.Module):
    def __init__(self, n_feature, n_class, alpha=0.1):
        super().__init__() 
        # 中心不需要求导，使用buffer，而非parameter
        self.register_buffer(
            "center", torch.zeros([n_class, n_feature]))
        # 指数加权滑动平均参数
        self.alpha = alpha
    def forward(self, h, d):
        """
        h: 表示向量
        d: 标签
        """
        # 回去每个类的均值
        batch_center = self.center[d] 
        # 计算中心损失
        loss_center = (h - batch_center).square().mean() 
        # 更新中心位置
        with torch.no_grad(): # 中心计算不需要梯度
            diff = torch.zeros_like(self.center) 
            for k in d: # 统计每个类别中心变化量
                diff[k] = (h[d==k]-self.center[k]).mean(dim=0)
            # 更新中心位置
            self.center += self.alpha * diff  
        return loss_center  



    
from plotconfig import * 
def main():
    dataset = ImagetDataset() 
    dataloader = DataLoader(dataset, 100, collate_fn=collate_batch)
    model = ImageClassify() 
    model.train() 
    lossfn = nn.CrossEntropyLoss() 
    lossce = CenterLoss(2, 19, 0.1)
    optim = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=0.00) 
    step = 0 
    try:
        model.load_state_dict(torch.load("ckpt/imageclassify.center"))
    except:
        pass 
    for e in range(100):
        for x, d in dataloader:
            y, h = model(x) 
            loss1 = lossfn(y, d) 
            loss2 = lossce(h, d)
            loss = loss1 + loss2 * 1
            loss.backward() 
            optim.step() 
            optim.zero_grad()
            step += 1 
            if step % 50 ==0:
                torch.save(model.state_dict(), "ckpt/imageclassify.center") 
                print(loss)
        fig = plt.figure(1, figsize=(9, 9)) 
        gs = grid.GridSpec(1, 1) 
        ax = fig.add_subplot(gs[0]) 
        h = h.detach().cpu().numpy() 
        d = d.detach().cpu().numpy() 
        names = ["飞机", "手机", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
        center = lossce.center
        center = center.cpu().numpy()
        for i in range(10):
            v = h[d==i]
            c = center[i]
            ax.scatter(v[:, 0], v[:, 1], marker=f"${i}$", s=200, c="k", alpha=0.5) 
            t = np.linspace(0, np.pi*2, 1000)
            r = (np.max(v)-np.min(v))/4
            x = c[0] + np.cos(t) * r 
            y = c[1] + np.sin(t) * r
            ax.plot(x, y, c="k", linestyle="--", lw=1.5, alpha=0.6)
            #ax.plot(x, y, c="k", linestyle="--", lw=1.5, alpha=0.6)
        #ax.plot([], [], c="k", linestyle="--", lw=1.5, alpha=0.6, label="不同类别")
        for i in range(10):
            ax.scatter([], [], marker=f"${i}$", s=20, c="k", alpha=0.5, label=names[i])  
        ax.legend(loc="upper right", ncol=2, fontsize=12) 
        max = np.max(np.abs(h)) 
        ax.set_xlim(([-max, max]))
        ax.set_ylim(([-max, max]))
        plt.savefig("导出图像/CIFAR分类.中心损失.1.jpg")
        plt.savefig("导出图像/CIFAR分类.中心损失.1.svg")
        plt.close()
        print(e, "绘图完成")



if __name__ == "__main__":
    main()