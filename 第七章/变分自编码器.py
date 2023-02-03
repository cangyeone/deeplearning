import pickle 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader 
import numpy as np 
from plotconfig import *
from 自编码器2 import AutoEncoder 

class ImagetDataset(Dataset):
    def __init__(self):
        file_ = np.load("data/mnist.npz")
        self.images = file_["x_train"]
        self.labels = file_["y_train"]
        #self.images = self.images[self.labels==8]
        #self.labels = self.labels[self.labels==8]
        x_test = file_["x_test"]
        d_test = file_["y_test"]
        #x_test = x_test[d_test==8] 
        #d_test = d_test[d_test==8]

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

class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__() 
        # 输入图形大小为32×32 
        # DNN输出长度为2的表示向量
        self.encoder = nn.Sequential(
            ConvBNReLU( 1, 16, 2), 
            ConvBNReLU(16, 32, 2), 
            ConvBNReLU(32, 64, 2), 
            ConvBNReLU(64, 128, 2), 
            ConvBNReLU(128, 256, 2), 
            nn.Conv2d(256, latent_dim*2, 1, 1),
        ) 
        # 构建生成模型：多层上采样+卷积
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 256, 1, 1),
            ConvUpBNReLU(256, 128, 2), 
            ConvUpBNReLU(128, 64, 2), 
            ConvUpBNReLU(64, 32, 2), 
            ConvUpBNReLU(32, 16, 2), 
            ConvUpBNReLU(16, 8, 2), 
            nn.Conv2d(8, 1, 1, 1), 
            nn.Sigmoid() # 输出[0,1]区间
        )        
        self.latent_dim = latent_dim 
    def forward(self, x, z=None):
        if z == None:# 训练过程中仅有z
            h = self.encoder(x)
            # 计算均值和方差对数，直接拟合方差有小于0情况
            mu = h[:, :self.latent_dim, :, :]     # 均值 
            logvar = h[:, self.latent_dim:, :, :] # log(方差)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            # 对隐变量进行采样
            s = eps * std + mu 
            y = self.decoder(s)
            return y, s, mu, logvar
        else:        # 推断过程需要输入z
            y = self.decoder(z)
            return y, z  

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
class KLLoss(nn.Module):
    def __init__(self):
        super().__init__() 
    def forward(self, mu, logvar):
        # 数据均值mu,logvar方差对数
        loss = torch.sum(
                -1-logvar + mu ** 2 + logvar.exp(), dim = 1
            ).mean()
        return loss 
            
from plotconfig import * 
def main():
    device = torch.device("cpu")
    dataset = ImagetDataset() 
    dataloader = DataLoader(dataset, 100, collate_fn=collate_batch)
    nhidden = 128
    model = VAE(nhidden) 
    optim = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=0.0) 
    lossce = nn.CrossEntropyLoss()
    lossmse = BCELoss()
    losskl = KLLoss()
    mse = nn.MSELoss()
    step = 0 
    model.to(device) 
    try:
        model.load_state_dict(torch.load("ckpt/vaes.pt", map_location="cpu"))
    except:
        pass 
    for e in range(1000):
        for x, d in dataloader:
            x = x.to(device) 
            y, h, mu, logvar = model(x)
            loss1 = mse(y, x)
            loss2 = losskl(mu, logvar)
            loss = loss1 + 0.01 * loss2 
            loss.backward() 
            optim.step() 
            optim.zero_grad() 
            step += 1
            N = len(x)
            if step % 50 ==0:
                torch.save(model.state_dict(), "ckpt/vae.pt") 
                print("绘图中", loss1, loss2)

                y = y.permute(0, 2, 3, 1).detach().cpu().numpy() 
                o = torch.tensor(np.linspace(0, np.pi*2-np.pi*2/8, 8)).float()
                r = torch.tensor(np.ones([8])).float()
                z = torch.randn([8, nhidden, 1, 1])
                #z[:, 0, 0, 0] = r * torch.cos(o) 
                #z[:, 1, 0, 0] = r * torch.sin(o) 
                c, _ = model(x, z)
                c = c.permute(0, 2, 3, 1).detach().cpu().numpy() 
                x = x.permute(0, 2, 3, 1).detach().cpu().numpy() 
                z = z.reshape(len(z), nhidden).detach().cpu().numpy() 
                h = h.reshape(N, nhidden).detach().cpu().numpy() 
                d = d.cpu().numpy()
                #names = ["飞机", "手机", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]

                #ax = fig.add_subplot(gs[0, 0]) 
                #ax.imshow(y[0, :, :, 0], cmap="Greys")
                #ax.set_xticks(()) 
                #ax.set_yticks(()) 
                #ax.set_title(f"(a) 训练数据生成{d[0]}")
                #ax = fig.add_subplot(gs[0, 1]) 
                #ax.imshow(c[0, :, :, 0], cmap="Greys")
                #ax.set_xticks(()) 
                #ax.set_yticks(()) 
                #ax.set_title("(b) 使用随机向量生成图像")
                #ax = fig.add_subplot(gs[1, 0]) 
                ##ax.hist(h[0], color="#000000", density=True)
                #for i in range(10):
                #    ax.scatter(h[d==i, 0], h[d==i, 1], c="k", marker=f"${i}$", s=50)
                #ax.set_title("(c) 变分向量散点")
                #ax = fig.add_subplot(gs[1, 1]) 
                #for i in range(10):
                #    ax.scatter(z[d==i, 0], z[d==i, 1], c="k", marker=f"${i}$", s=50)
                #ax.set_title("(d) 随机向量散点图")
                fig = plt.figure(1, figsize=(12, 12)) 
                gs = grid.GridSpec(3, 3) 
                for i in range(3):
                    for j in range(3):
                        idx = i * 3 + j - 1 
                        ax = fig.add_subplot(gs[i, j]) 
                        if idx < 0:
                            for k, (a, b) in enumerate(zip(z[:, 0], z[:, 1])):
                                ax.scatter(z[k:k+1, 0], z[k:k+1, 1], c="k", marker=f"${k}$") 
                                ax.set_title("a)") 
                        else:
                            ax.matshow(c[idx, :, :, 0], cmap="Greys") 
                            ax.set_title(f"隐变量{idx}生成图像")  
                        ax.set_xticks(()) 
                        ax.set_yticks(())                            
                plt.savefig("导出图像/vae.mnist.t2.png")
                plt.savefig("导出图像/vae.mnist.t2.svg")
                plt.close()
                print(e, "绘图完成")



if __name__ == "__main__":
    main()