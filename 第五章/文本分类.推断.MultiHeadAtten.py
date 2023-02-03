from plotconfig import * 
import numpy as np 

import torch.nn as nn
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
            self.train_text_id.append([self.word2id[w] for w in x[:20]]+[0])
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

import math 
class PositionalEncoding(nn.Module):
    # 位置编码类
    def __init__(self,
                 emb_size: int,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        # 三角函数
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        # 书中的公式
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        # 不可训练，因此写为Buffer
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        # 位置编码为加法
        return token_embedding + self.pos_embedding[:token_embedding.size(0), :]

class Model(nn.Module):
    def __init__(self, n_word, n_class):
        super().__init__() 
        n_hidden = 64
        self.pos = PositionalEncoding(n_hidden)
        # 文本向量化
        self.emb = nn.Embedding(n_word, n_hidden)
        # 多头注意力机制
        self.ma = nn.MultiheadAttention(n_hidden, 2)
        # 分类模型
        self.cls = nn.Linear(n_hidden, n_class)
        self.n_hidden = n_hidden
    def forward(self, x):
        x = self.emb(x) 
        x = self.pos(x) # 位置编码
        T, B, C = x.shape 
        # 按顺序将时间向量输入
        y, att = self.ma(x, x, x, attn_mask=maks)# y为最后一层输出，hT是两层状态向量
        # 最后一个时间步包含全文信息，用于分类
        y = self.cls(y[-1])
        return y, att 




textdata = TextDataset() 
dataloader = DataLoader(textdata, 20, shuffle=True, collate_fn=collate_fn)
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

model = Model(len(textdata.word2id), len(textdata.label2id)) 
model.eval() 
device = torch.device("cpu")
model.load_state_dict(torch.load("ckpt/text_classify_multihead.pt", map_location=device))


batch_size = 100

i2w = {} 
for key in textdata.word2id:
    i2w[textdata.word2id[key]] = key 

from plotconfig import * 

gs = grid.GridSpec(1, 1) 
fig = plt.figure(1, figsize=(18, 6))
ax = fig.add_subplot(gs[0])
for e in range(1):
    for x, d in dataloader:
        with torch.no_grad():
            y, att = model(x)
            x1 = x.cpu().numpy()
            p1 = y.argmax(dim=1).cpu().numpy()
            d1 = d.cpu().numpy()
            a2 = att.cpu().numpy() 
        print(a2.shape, x1.shape)
        im = ax.matshow(a2[:5, -1, :15])
        for b in range(5):
            for t in range(15):
                w = i2w[x1[t, b]]
                ax.text(t-0.4, b-0.4, w, va="top", ha="left", fontsize=36)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("时间步")
        ax.set_ylabel("样本编号")
        plt.savefig("导出图像/注意力2.svg")
        plt.savefig("导出图像/注意力2.pdf")
        plt.show()
        break 
    break 
        
