import pickle 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader 
import numpy as np 
from plotconfig import * 

class ImagetDataset(Dataset):
    def __init__(self):
        file_ = np.load("data/mnist.npz")
        self.images = file_["x_train"]
        self.labels = file_["y_train"]
        x_test = file_["x_test"]
        d_test = file_["y_test"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x, d = self.images[idx], self.labels[idx] 
        x = torch.tensor(x, dtype=torch.float32)/ 255 
        d = torch.tensor([d], dtype=torch.long)
        k = x.reshape([1, 28, 28])
        x = torch.zeros([1, 32, 32])
        x[:, 2:2+28, 2:2+28] = k 
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
        #self.emb = nn.Embedding(nclass, 128)
        #self.noize = nn.Linear(100, 128) 
        self.inputs = nn.Linear(110, 256)
        # 构建生成模型：多层上采样+卷积
        self.layers = nn.Sequential(
            ConvBNReLU(256, 256, 1), 
            ConvUpBNReLU(256, 128, 2), 
            ConvBNReLU(128, 128, 1), 
            ConvUpBNReLU(128, 64, 2), 
            ConvBNReLU(64, 64, 1), 
            ConvUpBNReLU(64, 32, 2), 
            ConvBNReLU(32, 32, 1), 
            ConvUpBNReLU(32, 16, 2), 
            ConvBNReLU(16, 16, 1), 
            ConvUpBNReLU(16, 8, 2), 
            nn.Conv2d(8, 1, 1, 1), 
            nn.Sigmoid() # 输出[0,1]区间
        )
    def forward(self, d, z):
        # 把类别信息进行编码
        h1 = F.one_hot(d, 10)
        h2 = z 
        h = torch.cat([h1, h2], dim=1) 
        h = self.inputs(h)
        h = h.reshape(-1, 256, 1, 1)
        x = self.layers(h)
        return x 

class ImageClassify(nn.Module):
    def __init__(self):
        super().__init__() 
        # DNN输出长度为2的表示向量
        self.dnn = nn.Sequential(
            ConvBNReLU( 1, 16, 2), 
            ConvBNReLU(16, 32, 2), 
            ConvBNReLU(32, 64, 2), 
            ConvBNReLU(64, 128, 2), 
            ConvBNReLU(128, 256, 2), 
            nn.Flatten(), 
            nn.Linear(256, 128), 
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
        #p = y.sigmoid() 
        #loss = -(d * torch.log(p) + (1-d)*torch.log(1-p))
        loss = (y-d)**2
        loss = loss.mean() 
        return loss 


from plotconfig import * 
def main():
    device = torch.device("cuda")
    dataset = ImagetDataset() 
    dataloader = DataLoader(dataset, 100, collate_fn=collate_batch)
    gen = ImageGeneration() 
    gen.train() 
    
    dis = ImageClassify() 
    dis.train() 
    optim_gen = torch.optim.Adam(gen.parameters(), 1e-3, weight_decay=0.0) 
    optim_dis = torch.optim.Adam(dis.parameters(), 1e-3, weight_decay=0.0) 
    lossce = nn.CrossEntropyLoss()
    lossmse = BCELoss()
    step = 0 
    gen.to(device) 
    dis.to(device)
    try:
        gen.load_state_dict(torch.load("ckpt/gen.acgan", map_location="cpu"))
        dis.load_state_dict(torch.load("ckpt/dis.acgan", map_location="cpu"))
    except:
        pass 
    for e in range(1000):
        for x, d in dataloader:
            x = x.to(device) 
            d = d.to(device)
            # 训练判别器 
            dis.zero_grad()
            # 生成随机数
            z = torch.randn([len(x), 100], device=device, dtype=torch.float32)
            # 生成图像
            fake = gen(d, z) 
            # 生成图像输入到判别器中
            y1, y2 = dis(fake.detach())
            # 真实图像输入到判别器中 
            y3, y4 = dis(x) 
            # 生成图像判别接近0
            loss1 = lossmse(y2, torch.zeros_like(y2))
            # 真实图像损失函数=类别损失+判别损失
            loss2 = lossce(y3, d) * 0.2 + lossmse(y4, torch.ones_like(y4))
            # 优化
            loss = loss1 + loss2 
            loss.backward()
            optim_dis.step() 
            optim_dis.zero_grad() 
            optim_gen.zero_grad() 
            
            # 训练判别器
            gen.zero_grad()
            z = torch.randn([len(x), 100], device=device, dtype=torch.float32)
            fake = gen(d, z) 
            y1, y2 = dis(fake)
            # 损失=类别损失+判别损失
            loss = lossce(y1, d) * 0.2 + lossmse(y2, torch.ones_like(y2)) 
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
        #names = ["飞机", "手机", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j 
                ax = fig.add_subplot(gs[i, j+4]) 
                ax.imshow(y[idx], cmap="Greys")
                ax.set_xticks(()) 
                ax.set_yticks(()) 
                ax.set_xlabel(f"生成:{d[idx]}")
                ax = fig.add_subplot(gs[i, j]) 
                ax.imshow(y[idx+50], cmap="Greys")
                ax.set_xticks(()) 
                ax.set_yticks(()) 
                ax.set_xlabel(f"生成:{d[idx+50]}")

        plt.savefig("导出图像/CIFAR生成.acgan.mnist.jpg")
        plt.savefig("导出图像/CIFAR生成.acgan.mnist.svg")
        plt.close()
        print(e, "绘图完成")



if __name__ == "__main__":
    main()