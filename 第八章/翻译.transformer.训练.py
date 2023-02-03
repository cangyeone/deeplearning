import torch
import torch.nn as nn
from torch.nn import Transformer
import math
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings

class TokenEmbedding(nn.Module):
    """文向量化类"""
    def __init__(self, n_wrods, emb_size):
        super(TokenEmbedding, self).__init__()
        # weizhi 
        self.embedding = nn.Embedding(n_wrods, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens):
        emb_size = self.emb_size 
        T, B  = tokens.shape 
        # # 此部分位置编码来自于《Attention is all you need 文章》
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos_embedding = torch.zeros((T, emb_size))
        # 利用三角函数是因为其中的周期性
        pos_embedding[:, 0::2] = torch.sin(T * den)
        pos_embedding[:, 1::2] = torch.cos(T * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        # 文本向量化
        word_embedding = self.embedding(tokens.long())
        return  pos_embedding + word_embedding 

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        #self.positional_encoding = PositionalEncoding(
        #    emb_size, dropout=dropout)

    def forward(self,
                src,
                trg,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                memory_key_padding_mask):
        src_emb = self.src_tok_emb(src)
        tgt_emb = self.tgt_tok_emb(trg)
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
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
    xs, ds, nmt, mk = [], [], [], []
    for x, d in batch:
        xs.append(x) 
        ds.append(d)
        nmt.append(len(x))
        mk.append(torch.ones_like(d))
    xs = rnnutils.pad_sequence(xs)
    mt = torch.zeros_like(xs, dtype=torch.float32)
    T, B = xs.shape 
    for itr in range(B):
        mt[nmt[itr]:, itr] += -1e6
    ds = rnnutils.pad_sequence(ds, padding_value=0)
    mk = rnnutils.pad_sequence(mk).float()
    return xs, ds, mk, mt  

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    """
    src:解码器输入
    tgt:编码器输入
    device:设备
    输出:输入掩码，注意力掩码
    """
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    # 输出数据的注意力掩码为上三角矩阵
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    # 输入数据的注意力掩码应当计算所有位置
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)
    # 补0的位置，这里补“0”的值为-1
    src_padding_mask = (src == -1).transpose(0, 1)
    tgt_padding_mask = (tgt == -1).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def main():
    train_dataset = TextDataset("data/cmn.txt")     
    train_dataloader = DataLoader(
        train_dataset, batch_size=2, 
        shuffle=True, collate_fn=collate_batch, num_workers=3)
    # 设定执行设备
    device = torch.device("cpu")
    
    embedding_size = 128 
    n_feature = 128 
    n_layer = 2 
    n_words_encoder = len(train_dataset.word2id) 
    n_words_decoder = len(train_dataset.word2id) 
    model = Seq2SeqTransformer(
        n_layer, 
        n_layer, 
        n_feature, 
        2, 
        n_words_encoder, 
        n_words_decoder, 
        n_feature
        )
    # 构建编码器和解码器
    model.train() 
    #model.load_state_dict(torch.load("ckpt/trans.pt"))
    model.to(device)
    optim = torch.optim.Adam(
        model.parameters(), 1e-4)
    # 定义损失函数
    lossfn = nn.CrossEntropyLoss(ignore_index=0)
    n_epoch = 100
    count = 0
    # 交叉熵作为损失函数
    for b in range(n_epoch):
        for x, d, m, s in train_dataloader:
            x = x.to(device) 
            d = d.to(device)
            m = m.to(device)
            s = s.to(device)
            d_input, d_target = d[:-1], d[1:] 
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask =\
                create_mask(x, d_input) 
            mask = m[1:]
            #print(x.shape, d_input.shape, src_mask.shape, tgt_mask.shape, src_padding_mask.shape, tgt_padding_mask.shape)
            y = model(x, d_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask)
            
            T, B, C = y.shape 
            d_target = d_target.reshape(-1) 
            y_seq = y.reshape(-1, C)
            loss = (lossfn(y_seq, d_target) * mask).mean()
            loss.backward() 
            optim.step()# 执行优化 
            optim.zero_grad()# 梯度置0
            count += 1
            if count % 50 ==0:
                # loss本身为计算图的一部分，需要detach将其分离出来
                print(b, count, loss.detach().cpu().numpy())
                # 保存模型
                torch.save(model.state_dict(), "ckpt/trans.pt")

if __name__ == "__main__":
    main()