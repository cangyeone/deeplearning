import pickle 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader 
import numpy as np 
from plotconfig import * 

class ImagetDataset(Dataset):
    def __init__(self):
        self.labels = [[] for i in range(10)] 
        self.images = [[] for i in range(10)]
        
        for i in range(5):
            with open(f"data/cifar-10-batches-py/data_batch_{i+1}", 'rb') as fo:
                data_dict = pickle.load(fo, encoding="bytes")
            d = data_dict[b"labels"]
            x = data_dict[b"data"]
            for a, b in zip(x, d):
                self.images[b].append(a)
        self.length = 10000000 
        for v in self.images:
            if len(v)<self.length:
                self.length = len(v)
    def __len__(self):

        return self.length

    def __getitem__(self, idx):
        collect = [] 
        c = np.random.randint(0, 10) 
        s1 = np.random.randint(0, len(self.images[c]))
        k = np.arange(10)
        k = np.random.choice(k[k!=c]) 
        s2 = np.random.randint(0, len(self.images[k]))
        x1 = torch.tensor(self.images[c][idx], dtype=torch.float32) / 128 - 1 
        x2 = torch.tensor(self.images[c][s1], dtype=torch.float32) / 128 - 1 
        x3 = torch.tensor(self.images[k][s2], dtype=torch.float32) / 128 - 1 
        
        x = torch.cat([x.reshape([1, 3, 32, 32]) for x in [x1, x2, x3]], dim=0)
        d = torch.tensor([c], dtype=torch.long)
        return x, d

def collate_batch(batch):
    """
    定义后处理函数
    """
    xs, ds = [], []
    for x, d in batch:
        xs.append(x) 
        ds.append(d)
    xs = torch.cat(xs, dim=0) 
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

class ImageClassify1(nn.Module):
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

class TripletLoss(nn.Module):
    def __init__(self, dist=0.5) -> None:
        super().__init__()
        # 计算间隔
        self.dist = dist 
    def forward(self, v1, v2, v3):
        """
        三元损失函数不需要标签
        样本数量为3*batch_size 
        """
        # v1v2相同类别，v3不同
        #v1 = F.normalize(v1, dim=1)
        #v2 = F.normalize(v2, dim=1)
        #v3 = F.normalize(v3, dim=1)
        loss = ((v1-v2)**2 + self.dist - (v1-v3)**2).sum(dim=1) 
        # 小于0 说明已经满足要求，损失函数变为0
        loss = F.relu(loss).mean()
        return loss


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
    dataloader = DataLoader(dataset, 100, collate_fn=collate_batch, shuffle=True)
    model = ImageClassify() 
    model.train() 
    print("样本数量", len(dataset))
    lossfn = nn.TripletMarginLoss()
    lossce = CenterLoss(2, 10, 0.1)
    optim = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=0.00) 
    step = 0 
    try:
        model.load_state_dict(torch.load("ckpt/imageclassify.trip"))
    except:
        pass 
    for e in range(1000):
        for x, d in dataloader:
            y, h = model(x) 
            #h = F.normalize(h, dim=1)
            #print(y.shape, h.shape, x.shape, d.shape)
            v1 = h[0::3]
            v2 = h[1::3]
            v3 = h[2::3]
            loss = lossfn(v1, v2, v3)# + lossce(v1, d) * 1 
            loss.backward() 
            optim.step() 
            optim.zero_grad()
            step += 1 
            if step % 50 ==0:
                torch.save(model.state_dict(), "ckpt/imageclassify.trip") 
                print(loss)
        fig = plt.figure(1, figsize=(9, 9)) 
        gs = grid.GridSpec(1, 1) 
        ax = fig.add_subplot(gs[0]) 
        h = v1.detach().cpu().numpy() 
        d = d.detach().cpu().numpy() 
        names = ["飞机", "手机", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]

        for i in range(10):
            v = h[d==i]
            ax.scatter(v[:, 0], v[:, 1], marker=f"${i}$", s=200, c="k", alpha=0.5, label=names[i]) 
            
        ax.legend(loc="upper right", ncol=2, fontsize=12) 
        max = np.max(h, axis=0)
        min = np.min(h, axis=0)
        ax.set_xlim(([min[0], max[0]]))
        ax.set_ylim(([min[1], max[1]]))
        plt.savefig("导出图像/CIFAR分类.trip.1.jpg")
        plt.savefig("导出图像/CIFAR分类.trip.1.svg")
        plt.close()
        print(e, "绘图完成")



if __name__ == "__main__":
    main()