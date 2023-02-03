from operator import pos
import torch 
import torch.nn as nn 
import math 
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 此部分位置编码来自于《Attention is all you need 文章》
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, emb_size))
        # 利用三角函数是因为其中的周期性
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        # 位置编码器不可训练，因此是buffer 
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return token_embedding + self.pos_embedding[:token_embedding.size(0), :]

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




class TransformerDecoderLayer(nn.Module):
    """
    Transformer 解码器层
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        # 多头注意力机制，用于处理输入
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 多头注意力机制，用于输入编码信息
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 前馈层
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), 
            nn.Linear(), #线性层
            nn.Dropout(), 
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask):
        # 本层用于处理输入信息
        # attn_mask注意力掩码用于屏蔽后文信息
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)# 残差层
        tgt = self.norm1(tgt) # 标准化
        # tgt为解码器输入
        # memory为编码器输出
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt



class TransformerEncoderLayer(nn.Module):
    """
    Transformer 模型编码器层
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, drop_out=0.1):
        super(TransformerEncoderLayer, self).__init__()
        # 模型中的自注意力层
        self.self_attn = nn.MultiheadAttention(d_model, nhead,  dropout=drop_out)
        # 前馈层
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), 
            nn.Linear(), #线性层
            nn.Dropout(), 
            nn.Linear(dim_feedforward, d_model)
        )
        # 两个层标准化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)

    def forward(self, src, src_mask, src_key_padding_mask):
        """
        src:输入
        src_mask:注意力掩码
        src_key_padding_mask:数据补0掩码
        """
        # 注意力层输出
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2) # 残差层
        src = self.norm1(src) # 标准化层
        # 前馈层
        src2 = self.feed_forward(src) 
        src = src + self.dropout1(src2) # 残差层
        src = self.norm2(src) #标准化层
        return src







class Transformer(nn.Module):
    """
    完整的Transformer模型
    编码器+解码器
    不包含文本向量化部分
    """
    def __init__(self, d_model, n_head=8, n_layer=6, drop_out=0.1):
        super(Transformer, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [
              TransformerEncoderLayer(d_model, n_head, drop_out=drop_out) \
                  for i in range(n_layer)
            ] # 定义编码器
        )
        self.decoder_layers = nn.ModuleList(
            [
              TransformerDecoderLayer(d_model, n_head, drop_out=drop_out) \
                  for i in range(n_layer)
            ] # 定义多层解码器
        )        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)        
        self.d_model = d_model
        self.n_head = n_head

    def forward(self, src, tgt, src_mask, tgt_mask, \
        memory_mask, src_key_padding_mask, tgt_key_padding_mask,\
             memory_key_padding_mask):
        enc_out = src 
        for enc_layer in self.encoder_layers:
            # src_mask用于控制注意力处理前后还是后文信息
            # src_key_padding_mask用于补零处理
            enc_out = enc_layer(enc_out, src_mask, src_key_padding_mask)
        enc_out = self.norm1(enc_out) # 编码器输出 

        memory = enc_out # 输入到解码器的是编码器输出
        dec_out = tgt # 解码器输出
        for dec_layer in self.decoder_layers:
            # tgt_mask用于控制输出仅包含前文信息
            # memory_mask用于控制编码器和解码器之间交互，用于解码器第二个多头注意力
            # tgt_key_padding_mask用于控制解码器之包含前文信息
            # memory_key_padding_mask用于处理编码器0部分的
            dec_out = dec_layer(dec_out, memory, tgt_mask, memory_mask,\
                tgt_key_padding_mask, memory_key_padding_mask)
        dec_out = self.norm2(dec_out)
        return dec_out 

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)



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

class Seq2Seq(nn.Module):
    def __init__(self, n_word):
        super().__init__() 
        # 编码器中由于处理STFT波形，因此不需要Embedding
        # 解码器中需要Embedding
        self.decoder_emb = nn.Embedding(n_word, 128)
        # 其他部分是一致的
        self.encoder = nn.GRU(128, 128, 2, bidirectional=False) 
        self.decoder = nn.GRU(128, 128, 2, bidirectional=False) 
        # 输出预测字符
        self.output = nn.Linear(128, n_word) 