import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss 
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader 
import torch.nn.utils.rnn as rnnutils
import os 
import numpy as np 


class TextDataset(Dataset):
    def __init__(self, file_name="data/cmn.txt"):
        if os.path.exists("ckpt/word2id.nmt"):
            with open("ckpt/word2id.nmt", "r", encoding="utf-8") as f:
                self.word2id = eval(f.read())
        else:
            file_ = open(file_name, "r", encoding="utf-8")
            words_set = set(file_.read()) 
            self.word2id = dict(zip(words_set, range(len(words_set))))
            with open("ckpt/word2id.nmt", "w", encoding="utf-8") as f:
                f.write(str(self.word2id))
            file_.close()
        label = [] 
        text = []
        file_ = open(file_name, "r", encoding="utf-8")
        for line in file_.readlines():
            sline = line.strip().split("\t")
            label.append(f"B{sline[0]}E") 
            text.append(sline[1])
    
        self.labels = [[self.word2id.get(i, 0) for n, i in enumerate(doc)] for doc in label] 
        self.inputs = [[self.word2id.get(i, 0) for n, i in enumerate(doc)] for doc in text] 
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
    xs, ds, mk = [], [], []
    for x, d in batch:
        xs.append(x) 
        ds.append(d)
        mk.append(torch.ones_like(d))
    xs = rnnutils.pack_sequence(xs, enforce_sorted=False)
    ds = rnnutils.pad_sequence(ds)
    mk = rnnutils.pad_sequence(mk).float()
    return xs, ds, mk 

class Encoder(nn.Module):
    def __init__(self, n_word):
        super().__init__()
        self.n_word = n_word
        self.n_hidden = 128 
        self.n_layer = 2 
        # 向量化函数
        self.emb = nn.Embedding(self.n_word, self.n_hidden)
        # 循环神经网络主体
        self.rnn = nn.GRU(self.n_hidden, self.n_hidden, self.n_layer)
        # 定义输出
    def forward(self, x, h0):
        x = self.emb(x) 
        y, ht = self.rnn(x, h0)
        return y, ht

class DecoderWithAtt(nn.Module):
    def __init__(self, n_words, n_hidden=128):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_words = n_words
        # 解码器需要嵌入文本
        self.emb = nn.Embedding(n_words, n_hidden)
        # 注意力机制
        self.wa = nn.Linear(n_hidden*2, n_hidden)
        self.va = nn.Linear(n_hidden, 1, False)
        # RNN模型解码器输入需要字符嵌入向量+注意力向量
        self.gru = nn.GRU(n_hidden*2, n_hidden, 2)
        # 将输出转换为字符数量
        self.out = nn.Linear(n_hidden, n_words)
        self.n_hidden = n_hidden 
    def forward(self, decoder_inputs, state, encoder_outputs):
        # 每次仅处理一个字符
        x = self.emb(decoder_inputs)
        T2, B, C = x.shape
        T1, B, C = encoder_outputs.shape 
        Att = torch.zeros([B, T1, T2]) 
        h = state 
        outputs = []
        for t2 in range(T2):
            xt = x[t2:t2+1]#[1, B, C]
            xr = xt.repeat(T1, 1, 1)#[T, B, C] 
            xr = torch.cat([xr, encoder_outputs], dim=2) 
            u = torch.tanh(self.wa(xr))
            e = self.va(u)
            score = F.softmax(e, dim=0) # [T, B, 1]
            Att[:, :, t2] = score.squeeze(dim=2).permute(1, 0)
            att_v = (encoder_outputs * score).sum(
                         0, keepdims=True)
            xt = torch.cat([xt, att_v], dim=2) 
            y, h = self.gru(xt, h)
            outputs.append(y)
        outputs = torch.cat(outputs, dim=0)
        y = self.out(outputs)
        return y, h, score   
from plotconfig import * 
def main():
    train_dataset = TextDataset("data/cmn.txt")     
    train_dataloader = DataLoader(
        train_dataset, batch_size=64, 
        shuffle=True, collate_fn=collate_batch, num_workers=3)
    # 设定执行设备
    device = torch.device("cpu")
    n_words_encoder = len(train_dataset.word2id) 
    n_words_decoder = len(train_dataset.word2id) 
    word2id = train_dataset.word2id 
    id2word = {} 
    for key in word2id:
        id2word[word2id[key]] = key 
    # 构建编码器和解码器
    encoder = Encoder(n_words_encoder) 
    decoder = DecoderWithAtt(n_words_decoder) 
    encoder.eval().to(device) 
    decoder.eval().to(device)
    encoder.load_state_dict(torch.load("ckpt/enc.att.pt", map_location=device))
    decoder.load_state_dict(torch.load("ckpt/dec.att.pt", map_location=device))

    text = "让我试试。"
    text_id = [word2id[w] for w in text] 
    outwords = []
    with torch.no_grad():
        x = torch.tensor(text_id, dtype=torch.long).unsqueeze(1) 
        h0 = torch.zeros(
            [2, 1, encoder.n_hidden]).to(device)
        encdoer_output, h_encoder = encoder(x.to(device), h0)
        # 解码器输入带开始标签的数据和编码器最终状态 
        h_decoder = torch.zeros(
            [2, 1, decoder.n_hidden]).to(device)
        crr_word = "B"
        att = []
        for itr in range(50):
            x = torch.tensor([[word2id.get(crr_word, 0)]], dtype=torch.long)
            y, h_decoder, attv = decoder(x.to(device), h_decoder, encdoer_output)
            pid = y.argmax(dim=2) 
            pid = pid.cpu().numpy()[0, 0] 
            crr_word = id2word.get(pid) 
            print(attv.shape, encdoer_output.shape)
            att.append(attv.numpy().reshape(1, -1))
            if crr_word == "E":break 
            outwords.append(crr_word)
        outwords = "".join(outwords)
        att = np.concatenate(att, axis=0)
        fig = plt.figure(1, figsize=(24, 8), dpi=100)
        gs = grid.GridSpec(1, 1) 
        ax = fig.add_subplot(gs[0, 0])
        CS = ax.matshow(att.T) 
        
        ax.set_xticks([i for i in range(len(outwords))])
        ax.set_xticklabels([i for i in outwords])
        ax.set_yticks([i for i in range(len(text))])
        ax.set_yticklabels([i for i in text])
        cbar = fig.colorbar(CS)
        cbar.ax.set_ylabel('注意力权值')
        # Add the contour line levels to the colorbar
        #cbar.add_lines(CS2)
        plt.savefig("导出图像/翻译注意力.svg")
        plt.savefig("导出图像/翻译注意力.jpg")
        plt.show()


           

if __name__ == "__main__":
    main()