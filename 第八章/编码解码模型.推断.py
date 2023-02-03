import os
from unicodedata import bidirectional 
from torch.utils.data import Dataset, DataLoader 
import torch 
from torch.nn.utils.rnn import pack_sequence, pad_sequence 
import torch.nn as nn 
class PoetryData(Dataset):
    def __init__(self, file_path="data/poetry.txt") -> None:
        super().__init__() 
        if os.path.exists("ckpt/poetry.w2i"):
            with open("ckpt/poetry.w2i", "r", encoding="utf-8") as f:
                self.word2id = eval(f.read())
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                words = f.read().replace("\n", "")
            swords = set(words) 
            self.word2id = dict(zip(swords, range(len(swords))))
            with open("ckpt/poetry.w2i", "w", encoding="utf-8") as f:
                f.write(str(self.word2id))    
        self.sid = len(self.word2id) 
        self.eid = len(self.word2id) + 1 
        self.datas = []
        # 给开始和结束标签一个独立ID
        self.word2id["<S>"] = len(self.word2id) 
        self.word2id["<E>"] = len(self.word2id) + 1 
        self.W = len(self.word2id)

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                sline = line.strip().split("，")
                if len(sline)!=2:continue 
                if len(sline[0])!=5 or len(sline[1])!=6:continue 
                self.datas.append(
                    [
                        [self.word2id[i] for i in sline[0]], 
                        [self.sid]+[self.word2id[i] for i in sline[1]]+[self.eid]
                    ]
                    
                )
    def __len__(self):
        return len(self.datas) 
    def __getitem__(self, index):
        sample = self.datas[index] 
        x  = torch.tensor(sample[0], dtype=torch.long) 
        d1 = torch.tensor(sample[1][:-1], dtype=torch.long)  
        d2 = torch.tensor(sample[1][1:], dtype=torch.long) 
        return (x, d1, d2) 

def collect_fn(batch):
    x1, d1, d2 = [], [], [] 
    for a, b, c in batch:
        x1.append(a) 
        d1.append(b) 
        d2.append(c) 
    #x1 = pack_sequence(x1, enforce_sorted=False) 
    x1 = pad_sequence(x1) 
    d1 = pad_sequence(d1) 
    d2 = pad_sequence(d2) 
    return x1, d1, d2 


class Seq2Seq(nn.Module):
    def __init__(self, n_word):
        super().__init__() 
        # 编码解码模型字符可能不同，但是诗句生成例子中是相同的
        # 但是为了程序通用性，这里使用不同的Embedding 
        self.encoder_emb = nn.Embedding(n_word, 128) 
        self.decoder_emb = nn.Embedding(n_word, 128)
        # 编码器两层
        # 由于需要传递状态向量，因此是单向模型
        self.encoder = nn.GRU(128, 128, 2, bidirectional=False) 
        # 严格来说编码器解码器可以是异构的
        # 但是对于词个实例来说由于需要状态向量，因此是同构的
        self.decoder = nn.GRU(128, 128, 2, bidirectional=False) 
        # 输出预测字符
        self.output = nn.Linear(128, n_word) 
    def forward(self, x, sid, eid):
        if self.training:# 是否是训练，通过model.train()/.eval()调整
            pass 
        else:
            xemb = self.encoder_emb(x)
            h0 = torch.zeros([2, 1, 128], device=x.device) 
            # y可以包含全文信息，状态向量同样可以
            y, hT = self.encoder(xemb, h0)
            # 将状态向量作为初始向量输入解码器中 
            yid = torch.ones([1, 1]).long() * sid 
            outputs = []
            for step in range(10):# 迭代最多10步 
                yemb = self.decoder_emb(yid) 
                y, hT = self.decoder(yemb, hT) 
                # 转换为字符概率，训练过程中不需要softmax处理
                y = self.output(y)
                yid = y.argmax(dim=2) 
                if yid[0, 0].numel() == eid:break 
                outputs.append(yid)
            outputs = torch.cat(outputs, dim=0)[:, 0]
        return outputs 



def main():
    dataset = PoetryData() 
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collect_fn)
    model = Seq2Seq(dataset.W) 
    model.eval() 
    model.load_state_dict(torch.load("ckpt/seq2seq.orig.pt", map_location="cpu"))
    lossfn = nn.CrossEntropyLoss(ignore_index=-1) # 补零的地方是-1不计入损失
    optim = torch.optim.Adam(model.parameters(), 1e-3) 
    n_iter = 0 
    id2word = {}
    for k, v in dataset.word2id.items():
        id2word[v] = k 
    for x, d1, d2 in dataloader:
        with torch.no_grad():
            y = model(x, dataset.sid, dataset.eid)    # d1带开始标签的序列
            y = y.cpu().numpy() 
            words1 = "".join([id2word[i] for i in x[:, 0].cpu().numpy()])
            words2 = "".join([id2word[i] for i in y])
            print(words1)
            print(words2)
         

if __name__ == "__main__":
    main()

                
