import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss 
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader 
from torch.nn.utils.rnn import pad_sequence
import os 
import numpy as np 


class TextDataset(Dataset):
    def __init__(self, file_name="data/reader.txt"):
        if os.path.exists("ckpt/word2id.reader"):
            with open("ckpt/word2id.reader", "r", encoding="utf-8") as f:
                self.word2id = eval(f.read())
        else:
            file_ = open(file_name, "r", encoding="utf-8")
            words_set = set(file_.read()) 
            self.word2id = dict(zip(words_set, range(len(words_set))))
            with open("ckpt/word2id.reader", "w", encoding="utf-8") as f:
                f.write(str(self.word2id))
            file_.close()
        label = [] 
        text = []
        file_ = open(file_name, "r", encoding="utf-8")
        for line in file_.readlines():
            line = line.strip().replace(" ", "")
            if len(line)<5:continue 
            label.append(line[1:]) 
            text.append(line[:-1])
        #file_text = file_.read() 
        #file_text = file_text.replace(" ", "").replace("\n", "。") 
        #file_text = file_text.split("。") 
        #for line in file_text:
        #    line = line.strip().replace(" ", "") + "。"
        #    if len(line)<3:continue 
        #    label.append(line[1:]) 
        #    text.append(line[:-1])            


        self.labels = [[self.word2id.get(i, 0) for i in doc] for doc in label] 
        self.inputs = [[self.word2id.get(i, 0) for i in doc] for doc in text] 
        self.n_word = len(self.word2id)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = (torch.Tensor(self.inputs[idx]).long(), torch.Tensor(self.labels[idx]).long())
        return sample

def collate_batch(batch):
    """
    定义后处理函数
    """
    xs, ds = [], []
    mk = []
    for x, d in batch:
        xs.append(x) 
        ds.append(d)
        mk.append(torch.ones_like(x))
    xs = pad_sequence(xs).long()
    ds = pad_sequence(ds).long()
    mk = pad_sequence(mk).float()
    return xs, ds, mk 

class TextGeneration(nn.Module):
    def __init__(self, n_word):
        super().__init__()
        self.n_word = n_word
        self.n_hidden = 512 
        self.n_layer = 2 
        # 文本向量化（Word Embedding）函数
        self.emb = nn.Embedding(self.n_word, self.n_hidden)
        # 循环神经网络主体（GRU网络）
        self.rnn = nn.GRU(self.n_hidden, self.n_hidden, self.n_layer)
        # 定义输出（变为字符类别预测）
        self.out = nn.Linear(self.n_hidden, self.n_word)
    def forward(self, x):
        B, T = x.shape 
        x = self.emb(x)
        # 定义初始状态向量
        h0 = torch.zeros(
            [self.rnn.num_layers, B, self.rnn.hidden_size], dtype=x.dtype)
        y, h0 = self.rnn(x, h0)
        y = self.out(y)
        return y 

model = TextGeneration()
optim = torch.optim.Adam(model.parameters(), 1e-3)
lossfn = CrossEntropyLoss()
for step in # 迭代过程
    x, d = # 训练数据，x:[T, B], D[T, B] 
    y = model(x) 
    # y:[T,B,C]->[T,C,B]
    loss = lossfn(y.permute(0, 2, 1), d)
    # 梯度下降过程