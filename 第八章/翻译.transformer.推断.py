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

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones(
        (sz, sz), device=device)) == 1).transpose(0, 1)
    # Attention为0，即不能包含后文信息，应当添加-inf
    mask = mask.float().masked_fill(
        mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    # 输出数据不能包含之后的信息
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    # 输入数据应当所有都可以计入统计
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)
    # 补0的位置
    src_padding_mask = (src == 0).transpose(0, 1)
    tgt_padding_mask = (tgt == 0).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def main():
    train_dataset = TextDataset("data/cmn.txt")     
    train_dataloader = DataLoader(
        train_dataset, batch_size=64, 
        shuffle=True, collate_fn=collate_batch, num_workers=3)
    # 设定执行设备
    device = torch.device("cuda")
    
    embedding_size = 128 
    n_feature = 128 
    n_layer = 2 
    n_words_encoder = len(train_dataset.word2id) 
    n_words_decoder = len(train_dataset.word2id) 
    word2id = train_dataset.word2id 
    id2word = {} 
    for key in word2id:
        id2word[word2id[key]] = key 
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
    model.eval() 
    model.load_state_dict(torch.load("ckpt/trans.pt"))
    model.to(device)
    text = "难以置信！"
    text_id = [word2id[w] for w in text] 
    outwords = []
    x = torch.tensor(text_id, dtype=torch.long).unsqueeze(1) 
    num_tokens = x.shape[0]
    x_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    x = x.to(device)
    x_mask = x_mask.to(device)

    memory = model.encode(x, x_mask)
    pre_word = "B"
    ys = torch.ones(1, 1).fill_(word2id.get(pre_word, 0)).type(torch.long).to(device)
    words = []
    for step in range(50):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device)
                    .type(torch.bool))
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(x.data).fill_(next_word)], dim=0)
        pre_word = id2word.get(next_word)
        if pre_word == "E":
            break
        words.append(pre_word)
    print("".join(words))
if __name__ == "__main__":
    main()