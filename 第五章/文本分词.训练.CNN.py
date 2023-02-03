import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss 
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader 
from torch.nn.utils.rnn import pad_sequence
import os 
import numpy as np 
import re 


class TextDataset(Dataset):
    def __init__(self, file_name="data/分词数据/pku_training.utf8"):
        
        if os.path.exists("ckpt/word2id.segment"):
            with open("ckpt/word2id.segment", "r", encoding="utf-8") as f:
                self.word2id = eval(f.read())
        else:
            file_ = open(file_name, "r", encoding="utf-8")
            words_set = set(file_.read()) 
            self.word2id = dict(zip(words_set, range(len(words_set))))
            with open("ckpt/word2id.segment", "w", encoding="utf-8") as f:
                f.write(str(self.word2id))
            file_.close()
        label = [] 
        text = []
        file_ = open(file_name, "r", encoding="utf-8")
        file_data = file_.read().replace("\n", " ") 
        file_data_seg = [i for i in file_data.split(" ") if len(i)>0]
        def make_label(txt):
            if len(txt) == 1:
                return "s" 
            elif len(txt) == 2:
                return "be" 
            else:
                return f"b{'m'*(len(txt)-2)}e"
        file_data_label = [make_label(i) for i in file_data_seg]
        x = "".join(file_data_seg) 
        d = "".join(file_data_label)
        L1 = len(x) 
        L2 = len(d) 
        assert L1 == L2 

        length = 30 
        L = L1//length 
        label = [] 
        text = []
        for i in range(L):
            text.append(x[i*length:i*length+length])
            label.append(d[i*length:i*length+length])
        self.id2word = {}
        for key in self.word2id:
            self.id2word[self.word2id[key]] = key 
        self.label2id = {"b":0, "m":1, "s":2, "e":3} 
        self.id2label = {0:"b", 1:"m", 2:"s", 3:"e"} 
        self.labels = [[self.label2id.get(i, 0) for n, i in enumerate(doc) if n<100] for doc in label] 
        self.inputs = [[self.word2id.get(i, 0) for n, i in enumerate(doc) if n<100] for doc in text] 
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
    for x, d in batch:
        xs.append(x) 
        ds.append(d)
    xs = pad_sequence(xs).long()
    ds = pad_sequence(ds).long()
    return xs, ds 

class WordSeg(nn.Module):
    def __init__(self, n_word):
        super().__init__()
        self.n_word = n_word
        self.n_hidden = 64 
        self.n_layer = 2 
        kernel_size = 5 # 卷积核心大小
        pad = (kernel_size-1)//2 # 卷积输入输出长度相同，加pad
        # 向量化函数
        self.emb = nn.Embedding(self.n_word, self.n_hidden)
        # 循环神经网络主体，bidirectional参数设置为True
        self.cnn = nn.Sequential(
            nn.Conv1d(self.n_hidden, self.n_hidden, 
                      kernel_size, padding=pad), 
            nn.ReLU(), 
            nn.Conv1d(self.n_hidden, self.n_hidden, 
                      kernel_size, padding=pad), 
            nn.ReLU(), 
            nn.Conv1d(self.n_hidden, self.n_hidden, 
                      kernel_size, padding=pad), 
            nn.ReLU()
        )
        # 定义输出，处理cnn输出BCT格式需要使用卷积
        self.out = nn.Conv1d(self.n_hidden, 4, 1)
    def forward(self, x):
        T, B = x.shape 
        x = self.emb(x)#T,B,C 
        x = x.permute(1, 2, 0) # 卷积神经网络需要B,C,T格式
        # 卷积神经网络
        h = self.cnn(x)
        y = self.out(h)
        return y 


def main():
    train_dataset = TextDataset("data/分词数据/pku_training.utf8")     
    train_dataloader = DataLoader(train_dataset, batch_size=40, shuffle=True, collate_fn=collate_batch, num_workers=3)

    gpu = False   #使用GPU 

    model = WordSeg(train_dataset.n_word)
    model.train() 
    if gpu:
        model.cuda() 
    #model.load_state_dict(torch.load("ckpt/wordseg.pt"))
    optim = torch.optim.Adam(model.parameters(), 1e-3)
    lossfn = CrossEntropyLoss()
    n_epoch = 100
    count = 0
    for b in range(n_epoch):
        for x, d in train_dataloader:
            t, nbatch = x.shape 
            if gpu:
                x = x.cuda()
                d = d.cuda()
            #x = x.permute(1, 0) 
            d = d.permute(1, 0)
            y = model(x) 
            # T, B, C->B, C, T 
            loss = lossfn(y, d) 
            loss.backward() 
            optim.step() 
            optim.zero_grad()
            count += 1
            # 今天 学习 的 是 深度循环神经网络
            if count % 50 ==0:
                p = y.detach().cpu().numpy() 
                p = np.argmax(p, axis=1)
                d = d.cpu().numpy()
                print(b, count, loss.detach().cpu().numpy())
                torch.save(model.state_dict(), "ckpt/wordseg.cnn.pt")

if __name__ == "__main__":
    main()