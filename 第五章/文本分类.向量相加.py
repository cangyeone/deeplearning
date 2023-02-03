from plotconfig import * 
import numpy as np 

import torch.nn as nn
import torch 
from torch.nn.utils.rnn import pad_sequence
class Model(nn.Module):
    def __init__(self, n_word, n_class):
        super().__init__() 
        n_hidden = 64
        # 文本向量化
        self.emb = nn.Embedding(n_word, n_hidden)
        # 用于分类
        self.lin = nn.Linear(n_hidden, n_class) 
    def forward(self, x):
        x = self.emb(x) 
        h = torch.mean(x, dim=0) # 将所有值进行相加
        y = self.lin(h)
        return y 



with open("ckpt/word2id", "r", encoding="utf-8") as f:
    word2id = eval(f.read())
with open("ckpt/label2id", "r", encoding="utf-8") as f:
    label2id = eval(f.read())
file_train = open("data/cnews.train.txt", "r", encoding="utf-8") 
train_text_id = [] 
train_label_id = []
for line in file_train.readlines():
    d, x = line.split("\t") #tab作为标签间隔
    train_text_id.append([word2id[w] for w in x[:50]])
    train_label_id.append(label2id[d])


test_text_id = [] 
test_label_id = []
file_test = open("data/cnews.test.txt", "r", encoding="utf-8") 
for line in file_test.readlines():
    d, x = line.split("\t") #tab作为标签间隔
    test_text_id.append([word2id.get(w, 0) for w in x[:50]])
    test_label_id.append(label2id[d])
x2 = [torch.tensor(w, dtype=torch.long) for w in test_text_id]
d2 = np.array(test_label_id)
x2 = pad_sequence(x2)

model = Model(len(word2id), len(label2id)) 
model.train() 
optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0) 
lossfn = nn.CrossEntropyLoss()

batch_size = 100
for e in range(10):
    for step in range(len(train_label_id)//batch_size-1):
        x = [torch.tensor(w, dtype=torch.long) for w in train_text_id[step*batch_size:(step+1)*batch_size]]
        d = torch.tensor(train_label_id[step*batch_size:(step+1)*batch_size], dtype=torch.long)
        x = pad_sequence(x)
        y = model(x)
        loss = lossfn(y, d) 
        loss.backward() 
        optim.step() 
        optim.zero_grad() 
        if step % 100 == 0:
            with torch.no_grad():
                p1 = y.argmax(dim=1).cpu().numpy()
                d1 = d.cpu().numpy()
                y2 = model(x2) 
                p2 = y2.argmax(dim=1).cpu().numpy() 
            torch.save(model.state_dict(), "ckpt/text_classify_sum.pt")
            print(f"{step},{np.mean(p1==d1)},{np.mean(p2==d2)}")
