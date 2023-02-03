import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss 
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader 
import torch.nn.utils.rnn as rnnutils
import os 
import numpy as np 


class TextDataset(Dataset):
    def __init__(self, file_name="data/poetry.txt"):
        
        if os.path.exists("ckpt/word2id.poetry"):
            with open("ckpt/word2id.poetry", "r", encoding="utf-8") as f:
                self.word2id = eval(f.read())
        else:
            file_ = open(file_name, "r", encoding="utf-8")
            words_set = set(file_.read()) 
            self.word2id = dict(zip(words_set, range(len(words_set))))
            with open("ckpt/word2id.poetry", "w", encoding="utf-8") as f:
                f.write(str(self.word2id))
            file_.close()
        label = [] 
        text = []
        file_ = open(file_name, "r", encoding="utf-8")
        for line in file_.readlines():
            line = line + "\n"
            label.append(line[1:]) 
            text.append(line[:-1])
    
        self.labels = [[self.word2id.get(i, 0) for n, i in enumerate(doc) if n<100] for doc in label] 
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
    xs = rnnutils.pack_sequence(xs, enforce_sorted=False)
    ds = rnnutils.pack_sequence(ds, enforce_sorted=False)
    return xs, ds 

import torch.nn.functional as F 
class CausalCNN(nn.Module):
    def __init__(self, nin, nout, kernel_size=3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            nin, nout, kernel_size, padding=0)
        self.pad = kernel_size - 1
    def forward(self, x):
        x = F.pad(x, [self.pad, 0])
        y = self.conv(x)
        return y 

class Model(nn.Module):
    def __init__(self, n_word):
        super().__init__()
        self.n_word = n_word
        self.n_hidden = 128 
        self.n_layer = 2 
        # 向量化函数
        self.emb = nn.Embedding(self.n_word, self.n_hidden)
        # 循环神经网络主体
        self.cnn = nn.Sequential(
            CausalCNN(self.n_hidden, self.n_hidden, 5), 
            nn.BatchNorm1d(self.n_hidden), 
            nn.ReLU(),
            #CausalCNN(self.n_hidden, self.n_hidden, 5), 
            #nn.BatchNorm1d(self.n_hidden), 
            #nn.ReLU(),
            #CausalCNN(self.n_hidden, self.n_hidden, 5), 
            #nn.BatchNorm1d(self.n_hidden), 
            #nn.ReLU(),
        )
        # 定义输出
        self.out = nn.Conv1d(self.n_hidden, self.n_word, 1)
    def forward(self, x):
        x = self.emb(x)
        x = x.permute(1, 2, 0)
        y = self.cnn(x)
        y = self.out(y)
        return y

def main():
    train_dataset = TextDataset("data/poetry.txt")     
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch, num_workers=3)

    device = torch.device("cuda")

    model = Model(train_dataset.n_word)
    model.eval() 
    model.load_state_dict(torch.load("ckpt/article.cnn2.pt"))
    #model = Model(len(word2id)) 
    #model.eval() # 调整为推断模型
    #model.load_state_dict(torch.load("ckpt/poetry.pt", map_location=torch.device("cpu")))
    pres = ['自', "然", "语", "言"] 
    #-------------------------------字典读取---------------------------#  
    with open("ckpt/word2id.poetry", "r", encoding="utf-8") as word_dict_file:
            word2id = eval(word_dict_file.read()) 
    id2word = {} 
    for key in word2id:
        id2word[word2id[key]] = key 
    def to_word(p):  
        #t = np.cumsum(weights)  
        #s = np.sum(weights)  
        #sample = int(np.searchsorted(t, np.random.random()*s))  
        wid = np.argmax(p)# 选择概率最大的词的ID 
        #wid = np.random.choice(len(p), p=p)
        return id2word[wid]  
    print(len(word2id))


    for s in pres:
        # 初始状态是0 
        words = [s]
        for i in range(15):# 最大解码15个字符
            w = torch.Tensor([[word2id[w]] for w in words]).long()
            y = model(w)#将字符ID输入,h输入获取输出。
            p = y.softmax(dim=1)[:, :, -1]
            p = p.detach().numpy()
            p = np.reshape(p, [-1]) 
            w = to_word(p)
            if w == "\n":# 这里使用\n作为结束标签
                break 
            words.append(w) 
        print("".join(words))

if __name__ == "__main__":
    main()