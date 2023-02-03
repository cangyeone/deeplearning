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


class LSTMCell(nn.Module):
    # 单层LSTM，以单个时间步作为输入
    def __init__(self, nin, nout):
        super().__init__() 
        # 3个门+1个输出=总共四个输出
        self.layer = nn.Linear(nin+nout, 4*nout)
        self.nout = nout 
    def forward(self, xt, state):
        # 某个时间步输入
        ht, ct = state 
        x = torch.cat([xt, ht], dim=1) 
        h = self.layer(x) 
        g1, g2, g3, o = torch.split(h, self.nout, dim=1) 
        g1 = g1.sigmoid() # 遗忘门
        g2 = g2.sigmoid() # 输入门
        g3 = g3.sigmoid() # 输出门 
        o = o.tanh()     # 输出
        ct1 = ct * g1 + o * g2 
        ht1 = ct1.tanh() * g3 
        return ht1, (ht1, ct1)

class LSTM(nn.Module):
    # 多层LSTM
    def __init__(self, nin, nout, nlayer):
        super().__init__() 
        self.layers = nn.ModuleList(
            [LSTMCell(nin, nout)] + \
     [LSTMCell(nout, nout) for i in range(nlayer-1)]
        )
    def forward(self, x, h):
        T, B, C = x.shape 
        ht, ct = h # 获取初始状态向量
        outputs = []
        # 每一层的状态向量 
        hts = [ht[i] for i in range(len(self.layers))]
        cts = [ct[i] for i in range(len(self.layers))]
        for t in range(T):
            xt = x[t]
            # 每个时间步依次输入多个层
            newht, newct = [], []
            for idx, layer in enumerate(self.layers):
                # 输入到idx层网络中
                xt, (ht_layer, ct_layer) = layer(xt, (hts[idx], cts[idx]))
                newht.append(ht_layer) 
                newct.append(ct_layer)
            hts = newht 
            cts = newct 
            outputs.append(xt) # 最后一层输出作为神经网络输出
        outputs = torch.stack(outputs, dim=0) # 时间步维度连接
        return outputs, (ht, ct) 


class Model(nn.Module):
    def __init__(self, n_word, n_class):
        super().__init__() 
        n_hidden = 64
        n_layer = 2 # 层数
        # 文本向量化
        self.emb = nn.Embedding(n_word, n_hidden)
        # 循环神经网络使用LSTM 
        self.rnn = LSTM(n_hidden, n_hidden, n_layer)
        # 分类模型
        self.cls = nn.Linear(n_hidden, n_class)
        self.n_hidden = n_hidden
        self.n_layer = n_layer 
    def forward(self, x):
        x = self.emb(x) 
        T, B, C = x.shape 
        # 初始状态向量和记忆向量为0, 有两层 
        h = torch.zeros([self.n_layer, B, C])
        c = torch.zeros([self.n_layer, B, C])
        # 按顺序将时间向量输入
        y, (hT, cT) = self.rnn(x, (h, c))# y为最后一层输出，hT是两层状态向量
        # 最后一个时间步包含全文信息，用于分类
        y = self.cls(x.mean(dim=0))
        return y 



textdata = TextDataset() 
dataloader = DataLoader(textdata, 32, shuffle=True, collate_fn=collate_fn)



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
model.train() 
optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0) 
lossfn = nn.CrossEntropyLoss()

batch_size = 100
for e in range(10):
    for x, d in dataloader:
        y = model(x)
        loss = lossfn(y, d) 
        loss.backward() 
        optim.step() 
        optim.zero_grad() 
    with torch.no_grad():        
        p1 = y.argmax(dim=1).cpu().numpy()
        d1 = d.cpu().numpy()
        y2 = model(x2) 
        p2 = y2.argmax(dim=1).cpu().numpy() 
    torch.save(model.state_dict(), "ckpt/text_classify_sum.pt")
    print(f"{e},{np.mean(p1==d1)},{np.mean(p2==d2)}")
