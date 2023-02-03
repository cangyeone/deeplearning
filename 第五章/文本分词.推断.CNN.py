import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss 
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader 
from torch.nn.utils.rnn import pad_sequence
import os 
import numpy as np 

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
        self.id2label = {0:"B", 1:"M", 2:"S", 3:"E"} 
        self.labels = [[self.label2id.get(i, 0) for n, i in enumerate(doc) if n<100] for doc in label] 
        self.inputs = [[self.word2id.get(i, 0) for n, i in enumerate(doc) if n<100] for doc in text] 
        self.n_word = len(self.word2id)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = (torch.Tensor(self.inputs[idx]).long(), torch.Tensor(self.labels[idx]).long())
        return sample


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
        x = self.emb(x)#T,B,C 
        x = x.permute(1, 2, 0) # 卷积神经网络需要B,C,T格式
        # 卷积神经网络
        h = self.cnn(x)
        y = self.out(h)
        return y 

def main():
    train_dataset = TextDataset("data/分词数据/pku_training.utf8")     
    gpu = False   #使用GPU 

    model = WordSeg(train_dataset.n_word)
    model.eval() 
    if gpu:
        model.cuda() 
    model.load_state_dict(torch.load("ckpt/wordseg.cnn.pt"))
    text = "本章学习循环神经网络。"
    textid = [[train_dataset.word2id.get(i, 0)] for i in text] 
    inputs = torch.Tensor(textid).long() 
    h0 = torch.zeros([2*2, 1, model.n_hidden]) 
    y = model(inputs) 
    y = y.detach().numpy() 
    ids = np.argmax(y, axis=1).reshape(-1) 
    print(text)
    print("".join([train_dataset.id2label[i] for i in ids]))

if __name__ == "__main__":
    main()