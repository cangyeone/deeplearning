from plotconfig import * 
import numpy as np 

import torch 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader 

class TextDataset(Dataset):
    def __init__(self) -> None:
        super().__init__() 
        with open("ckpt/word2id", "r", encoding="utf-8") as f:
            self.word2id = eval(f.read())
        with open("ckpt/label2id", "r", encoding="utf-8") as f:
            self.label2id = eval(f.read())
        file_train = open("data/cnews.train.txt", "r", encoding="utf-8") 
        self.train_text_id = [] 
        self.train_label_id = []
        for line in file_train.readlines():
            d, x = line.split("\t") #tab作为标签间隔
            self.train_text_id.append([self.word2id[w] for w in x[:50]])
            self.train_label_id.append(self.label2id[d])
    def __len__(self):
        return len(self.train_label_id)
    def __getitem__(self, index):
        x = torch.tensor(self.train_text_id[index], dtype=torch.long) 
        d = torch.tensor([self.train_label_id[index]], dtype=torch.long) 
        return (x, d) 
def collate_fn(batch):
    xs, ds = [], []
    for x, d in batch:
        xs.append(x) 
        ds.append(d) 
    xs = pad_sequence(xs)
    ds = torch.cat(ds) 
    return (xs, ds)  

import mynn.nn as nn 
import mynn 
import mynn.nn.functional as F 
class Model(nn.Module):
    def __init__(self, n_word, n_class):
        super().__init__() 
        n_hidden = 32
        # 文本向量化
        self.emb = nn.Embedding(n_word, n_hidden)
        # 循环神经网络使用的全连接层 
        self.rnn_base = nn.Sequential(
            nn.Linear(n_hidden+n_hidden, n_hidden), 
            nn.Tanh() # 选择有上下边界（饱和）的激活函数
        )
        # 分类模型
        self.cls = nn.Linear(n_hidden, n_class)
        self.n_hidden = n_hidden
    def forward(self, x):
        x = self.emb(x) 
        T, B, C = x.shape 
        # 初始状态向量为0 
        h = mynn.zeros([B, C])
        # 按顺序将时间向量输入
        for t in range(T):
            xt = x[t]
            #print(xt.shape)
            xt = mynn.cat([xt, h], dim=1)
            h = self.rnn_base(xt) 
            #print("正向", t, h.data.max(), h.data.min(), np.abs(xt.data).max())
            #plt.matshow(h.data) 
            #plt.show()
        # 最后一个时间步包含全文信息，用于分类
        y = self.cls(h)
        return y 




textdata = TextDataset() 
dataloader = DataLoader(textdata, 30, shuffle=True, collate_fn=collate_fn)
test_text_id = [] 
test_label_id = []
file_test = open("data/cnews.test.txt", "r", encoding="utf-8") 
for line in file_test.readlines():
    d, x = line.split("\t") #tab作为标签间隔
    test_text_id.append([textdata.word2id.get(w, 0) for w in x[:50]])
    test_label_id.append(textdata.label2id[d])
x2 = [torch.tensor(w, dtype=torch.long) for w in test_text_id]
d2 = np.array(test_label_id)
x2 = pad_sequence(x2)

x2 = x2.numpy()[:200]
d2 = d2[:200]

model = Model(len(textdata.word2id), len(textdata.label2id)) 
#model.train() 
print("可训练参数数量", len(model.parameters()))
for var in model.parameters():
    print(var.shape)

optim = mynn.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0) 
lossfn = nn.CrossEntropyLoss()
batch_size = 100
n= 0 
for e in range(10):
    for x, d in dataloader:
        x = mynn.Tensor(x.numpy()) 
        d = mynn.Tensor(d.numpy())
        y = model(x)
        loss = lossfn(y, d) 
        loss.backward() 
        optim.step() 
        optim.zero_grad()   
        if n % 10 ==0:
            p1 = y.data.argmax(axis=1)
            d1 = d.data 
            #x2 = mynn.Tensor(x2) 
            #y2 = model(x2)
            #p2 = y2.data.argmax(axis=-1)
            #print(p2.shape, y2.shape)
            print(f"{e},{np.mean(p1==d1)}")
        n+=1 
    #d1 = d.cpu().numpy()
    #y2 = model(x2) 
    #p2 = y2.argmax(dim=1).cpu().numpy() 
    #torch.save(model.state_dict(), "ckpt/text_classify_basic_rnn.pt")
    #print(f"{e},{np.mean(p1==d1)},{np.mean(p2==d2)}")