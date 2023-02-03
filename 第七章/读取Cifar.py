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
            x = data_dict[b"labels"]
            d = data_dict[b"data"]
            self.labels.extend(d) 
            self.images.extend(x)

    def __len__(self):
        return len(self.images)

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
with open("data/cifar-10-batches-py/data_batch_1", 'rb') as fo:
    data_dict = pickle.load(fo, encoding="bytes")
labels = data_dict[b"labels"]
images = data_dict[b"data"]
print(len(images[0]))
fig = plt.figure(1, figsize=(18, 9), dpi=100) 
gs = grid.GridSpec(4, 8) 
names = ["飞机", "手机", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
for i in range(4):
    for j in range(8):
        ax = fig.add_subplot(gs[i, j]) 
        idx = i * 8 + j 
        ax.imshow(np.reshape(images[idx], [3, 32, 32]).transpose(1, 2, 0)) 
        ax.set_xticks(()) 
        ax.set_yticks(())
        ax.set_xlabel(f"{names[labels[idx]]}")
        #ax.set_xlabel(f"${names[labels[idx]]}$")

plt.savefig("导出图像/cifiar.svg")
plt.savefig("导出图像/cifiar.jpg")