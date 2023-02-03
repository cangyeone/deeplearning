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

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__() 
        # DNN输出长度为2的表示向量
        self.encoder = nn.Sequential(
            ConvBNReLU( 1, 16, 2), 
            ConvBNReLU(16, 32, 2), 
            ConvBNReLU(32, 64, 2), 
            ConvBNReLU(64, 128, 2), 
            ConvBNReLU(128, 256, 2), 
        ) 
        # 构建生成模型：多层上采样+卷积
        self.decoder = nn.Sequential(
            ConvUpBNReLU(256, 128, 2), 
            ConvUpBNReLU(128, 64, 2), 
            ConvUpBNReLU(64, 32, 2), 
            ConvUpBNReLU(32, 16, 2), 
            ConvUpBNReLU(16, 8, 2), 
            nn.Conv2d(8, 1, 1, 1), 
            nn.Sigmoid() # 输出[0,1]区间
        )        
    def forward(self, x, z=None):
        if z == None:
            h = self.encoder(x)
            y = self.decoder(h)
            return y 
        else:
            y = self.decoder(z)
            return y 

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
    device = torch.device("cpu")
    dataset = ImagetDataset() 
    dataloader = DataLoader(dataset, 100, collate_fn=collate_batch)
    model = AutoEncoder() 
    optim = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=0.0) 
    lossce = nn.CrossEntropyLoss()
    lossmse = BCELoss()
    mse = nn.MSELoss()
    step = 0 
    model.to(device) 
    try:
        model.load_state_dict(torch.load("ckpt/ae.pt", map_location="cpu"))
    except:
        pass 
    for e in range(1000):
        for x, d in dataloader:
            x = x.to(device) 
            y = model(x)
            loss = mse(y, x)
            loss.backward()
            optim.step() 
            optim.zero_grad() 
            step += 1
            if step % 50 ==0:
                torch.save(model.state_dict(), "ckpt/ae.pt") 
                print("绘图中")
                fig = plt.figure(1, figsize=(12, 12)) 
                gs = grid.GridSpec(2, 2) 
                y = y.permute(0, 2, 3, 1).detach().cpu().numpy() 
                z = torch.randn([100, 256, 1, 1])
                d = model(x, z)
                d = d.permute(0, 2, 3, 1).detach().cpu().numpy() 
                x = x.permute(0, 2, 3, 1).detach().cpu().numpy() 
                #names = ["飞机", "手机", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]

                ax = fig.add_subplot(gs[0, 0]) 
                ax.imshow(d[0, :, :, 0], cmap="Greys")
                ax.set_xticks(()) 
                ax.set_yticks(()) 
                ax.set_title("原始图像")
                ax = fig.add_subplot(gs[0, 1]) 
                ax.imshow(y[0, :, :, 0], cmap="Greys")
                ax.set_xticks(()) 
                ax.set_yticks(()) 

                plt.savefig("导出图像/ae.mnist.png")
                plt.savefig("导出图像/ae.mnist.svg")
                plt.close()
                print(e, "绘图完成")



if __name__ == "__main__":
    main()