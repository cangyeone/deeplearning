import torch 
import torch.nn as nn 
from utils.wider import ImagetDataset 

class Conv2d(nn.Module):
    def __init__(self, nin=8, nout=11, ks=3, st=1, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(nin, nout, ks, st, padding=padding), 
            nn.BatchNorm2d(nout), 
            nn.ReLU()
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 

class FaceDetection(nn.Module):
    # 全卷积神经网络，不包含全连接层
    def __init__(self):
        super().__init__() 
        self.layers = nn.Sequential(
            Conv2d(3, 32, 3, 2, padding=0), 
            Conv2d(32, 32, 3, 1, padding=0), 
            Conv2d(32, 64, 3, 2, padding=0), 
            Conv2d(64, 64, 3, 1, padding=0), 
            Conv2d(64, 64, 2, 1, 0), 
        )
        self.bbox = nn.Conv2d(64, 4, 1) 
        self.clas = nn.Conv2d(64, 2, 1)
    def forward(self, x):
        x = self.layers(x) 
        box = self.bbox(x) 
        cls = self.clas(x)
        return cls, box 

class FCNLoss(nn.Module):
    def __init__(self):
        super().__init__() 
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1) 
    def forward(self, cls, box, d1, d2):
        cls = cls.squeeze() 
        box = box.squeeze()
        loss1 = self.cross_entropy(cls, d1) 
        mask = (d1!=0).float()
        loss2 = (((box-d2) ** 2).mean(dim=1) * mask).mean()
        loss = loss1 + loss2 
        return loss 

def main():
    data_tool = ImagetDataset()
    device = torch.device("cpu")
    model = FaceDetection() 
    lossfn = FCNLoss() 
    model.train()
    lossfn.train() 
    try:
        model.load_state_dict(torch.load("ckpt/fcn.pt", map_location="cpu"))
    except:
        print("模型文件有问题,重新进行训练")
    model.to(device) 
    lossfn.to(device)
    optim = torch.optim.Adam(model.parameters(), 1e-3)
    for step in range(10000): 
        imgs, labs, boxes = data_tool.batch_data(100) 
        x = torch.tensor(imgs, dtype=torch.float32) / 255 
        d = torch.tensor(labs, dtype=torch.long) 
        b = torch.tensor(boxes, dtype=torch.float32) 
        x = x.permute(0, 3, 1, 2) 
        cls, box = model(x)       
        loss = lossfn(cls, box, d, b) 
        loss.backward()
        optim.step() 
        optim.zero_grad() 
        if step % 100 == 0:
            torch.save(model.state_dict(), "ckpt/fcn.pt") 
            print(step, loss)

if __name__ == "__main__":
    main()