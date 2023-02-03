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

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__() 
        pass 
    def forward(self, y, d):
        p = y.sigmoid() 
        loss = -(d * torch.log(p) + (1-d)*torch.log(1-p))
        loss = loss.mean() 
        return loss 


from plotconfig import * 
def main():
    device = torch.device("cpu")
    dataset = ImagetDataset() 
    dataloader = DataLoader(dataset, 100, collate_fn=collate_batch)
    gen = ImageGeneration() 
    gen.train() 
    optim_gen = torch.optim.Adam(gen.parameters(), 1e-3, weight_decay=0.0) 
    dis = ImageClassify() 
    dis.train() 
    optim_dis = torch.optim.Adam(dis.parameters(), 1e-3, weight_decay=0.0) 
    lossce = nn.CrossEntropyLoss()
    lossmse = BCELoss()
    step = 0 
    try:
        gen.load_state_dict(torch.load("gen.acgan"))
        dis.load_state_dict(torch.load("dis.acgan"))
    except:
        pass 
    for e in range(10):
        for x, d in dataloader:
            # 训练判别器 
            z = torch.randn([len(x), 100], device=device, dtype=torch.float32)
            fake = gen(d, z) 
            y1, y2 = dis(fake.detach()) 
            y3, y4 = dis(x) 
            loss1 = lossmse(y2.sigmoid(), torch.zeros_like(y2))
            loss2 = lossce(y3, d) * 10 + lossmse(y4.sigmoid(), torch.ones_like(y4))
            loss = loss1 + loss2 
            loss.backward()
            optim_dis.step() 
            optim_dis.zero_grad() 
            optim_gen.zero_grad() 

            z = torch.randn([len(x), 100], device=device, dtype=torch.float32)
            fake = gen(d, z) 
            y1, y2 = dis(fake)
            loss = lossce(y1, d) * 10 + lossmse(y2.sigmoid(), torch.ones_like(y2)) 
            loss.backward()
            optim_gen.step() 
            optim_gen.zero_grad() 
            optim_dis.zero_grad()
            step += 1
            if step % 50 ==0:
                torch.save(gen.state_dict(), "ckpt/gen.acgan") 
                torch.save(dis.state_dict(), "ckpt/dis.acgan") 
                print(loss, loss1, loss2)
        print("绘图中")
        fig = plt.figure(1, figsize=(16, 8)) 
        gs = grid.GridSpec(4, 8) 
        y = fake.permute(0, 2, 3, 1).detach().cpu().numpy() 
        x = x.permute(0, 2, 3, 1).detach().cpu().numpy() 
        d = d.detach().cpu().numpy() 
        names = ["飞机", "手机", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j 
                ax = fig.add_subplot(gs[i, j+4]) 
                ax.imshow(y[idx])
                ax.set_xticks(()) 
                ax.set_yticks(()) 
                ax.set_xlabel(f"生成:{names[d[idx]]}")
                ax = fig.add_subplot(gs[i, j]) 
                ax.imshow(x[idx])
                ax.set_xticks(()) 
                ax.set_yticks(()) 
                ax.set_xlabel(f"原始:{names[d[idx]]}")

        plt.savefig("导出图像/CIFAR生成.acgan.jpg")
        plt.savefig("导出图像/CIFAR生成.acgan.svg")
        plt.close()
        print(e, "绘图完成")



if __name__ == "__main__":
    main()