#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
RNN网络做自然语言处理
此文件用于神经网络模型的预测工作。
直接运行生成文本
"""

from unicodedata import bidirectional
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss 
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader 
from torch.nn.utils.rnn import pad_sequence
from torchaudio.transforms import MFCC 
import os 
import numpy as np 
import torchaudio 
from scipy.io import wavfile
import tqdm 


class Model(nn.Module):
    def __init__(self, n_word):
        super().__init__()
        self.n_word = n_word
        self.n_hidden = 128 
        self.n_layer = 2 
        self.norm = nn.BatchNorm1d(36)
        # 循环神经网络主体
        self.rnn = nn.GRU(36, self.n_hidden, self.n_layer, bidirectional=True)
        # 定义输出
        self.out = nn.Conv1d(self.n_hidden * 2, self.n_word, 1)
    def forward(self, x, h0):
        x = x.permute(1, 2, 0) 
        x = self.norm(x) 
        x = x.permute(2, 0, 1)
        y, h0 = self.rnn(x, h0)
        y = y.permute(1, 2, 0) # Channel第二dim
        y = self.out(y) 
        y = y.permute(2, 0, 1)
        return y 
with open("ckpt/word2id.speech", "r", encoding="utf-8") as f:
   word2id = eval(f.read())
model = Model(len(word2id)+1)
model.eval()
model.load_state_dict(torch.load("ckpt/speech.brnn.pt", map_location=torch.device('cpu')))

mfcc = MFCC(sample_rate=16000, n_mfcc=36) 


sr, wav = wavfile.read("data/data_thchs30/A2_0.wav")
wav = torch.from_numpy(wav).float().unsqueeze(0)
feq = mfcc(wav).permute(2, 0, 1) 
h0 = torch.zeros([2*2, 1, model.n_hidden])
y = model(feq, h0) 
y = torch.argmax(y, 2)
y = y.detach().cpu().numpy() 
y = np.reshape(y, [-1]) 

id2word = {}
for key in word2id:
    id2word[word2id[key]] = key 
blank = len(word2id)
i_pre = blank 
words = []
for i in y:
    if i==i_pre:
        continue 
    else:
        if i!=blank:
            words.append(id2word[i])
        i_pre = i
print("标签：", open("data/data_thchs30/A2_0.wav.trn", "r", encoding="utf-8").readline().replace(" ", "")) 
print("预测：", "".join(words))


