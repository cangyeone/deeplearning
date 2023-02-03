
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss 
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader 
from torch.nn.utils.rnn import pad_sequence
import os 
import numpy as np 
from scipy.io import wavfile
import scipy.signal as signal 
import tqdm 

class TextDataset(Dataset):
    def __init__(self, file_dir="data/data_thchs30"):
        if os.path.exists("data/wavchar.npz"):
            file_ = np.load("data/wavchar.npz")
            wavfiles = file_["wavfiles"] 
            wavlabel = file_["wavlabel"]
        else:
            file_names = os.listdir(file_dir)
            wavfiles = [] 
            wavlabel_files = []
            for itr in file_names:
                if itr.endswith(".wav"):
                    name = itr.split(".")[0]
                    wavfiles.append(os.path.join(file_dir, itr))
                    wavlabel_files.append(os.path.join(file_dir, f"{name}.wav.trn"))
            wavlabel = []
            for itr in tqdm.tqdm(wavlabel_files):
                with open(itr, "r", encoding="utf-8", errors="ignore") as f:
                    f.readline()
                    wavlabel.append(f.readline().strip().replace(" ", ""))
            np.savez("data/wavchar.npz", wavfiles=wavfiles, wavlabel=wavlabel)
        
        if os.path.exists("ckpt/word2id.char.speech"):
            with open("ckpt/word2id.char.speech", "r", encoding="utf-8") as f:
                self.word2id = eval(f.read())
        else:
            words = "".join(wavlabel)
            words_set = set(words) 
            self.word2id = dict(zip(words_set, range(len(words_set))))
            with open("ckpt/word2id.char.speech", "w", encoding="utf-8") as f:
                f.write(str(self.word2id))
        self.labels = [[self.word2id.get(i, 0) for n, i in enumerate(doc) if n<100] for doc in wavlabel] 
        self.wavfiles = wavfiles
        self.n_word = len(self.word2id)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        names = self.wavfiles[idx]
        sr, wav = wavfile.read(names)
        a, b, f = signal.stft(wav, fs=16000, nperseg=512, noverlap=256)
        #f = torch.stft(wav, 1024, 512)
        feq = np.abs(f).astype(np.float32).T 
        feq = torch.from_numpy(feq).float() 
        feq -= feq.mean() 
        feq /= (feq.std() + 1e-6)
        lab = torch.Tensor(self.labels[idx]).long() 
        lf = feq.shape[0] 
        ll = lab.shape[0]

        sample = (feq, lf, lab, ll)
        return sample

def collate_batch(batch):
    """
    定义后处理函数
    """
    xs, ds = [], []
    nx, nd = [], []
    for x, lx, d, ld in batch:
        xs.append(x) 
        ds.append(d)
        nx.append(lx) 
        nd.append(ld)
    xs = pad_sequence(xs).float()
    ds = pad_sequence(ds, batch_first=True).long() 
    nx = torch.Tensor(nx).long()
    nd = torch.Tensor(nd).long()
    return xs, ds, nx, nd

class Model(nn.Module):
    def __init__(self, n_word):
        super().__init__()
        self.n_word = n_word
        self.n_hidden = 128 
        self.n_layer = 2 
        # 循环神经网络主体
        self.rnn = nn.GRU(257, self.n_hidden, self.n_layer, bidirectional=False)
        # 定义输出
        self.out = nn.Conv1d(self.n_hidden, self.n_word, 1)
    def forward(self, x, h0):
        y, h0 = self.rnn(x, h0)
        y = y.permute(1, 2, 0) # Channel第二dim
        y = self.out(y) 
        y = y.permute(2, 0, 1)
        return y 

def main():
    train_dataset = TextDataset("data/data_thchs30")     
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch, num_workers=6)

    gpu = True  #使用GPU 

    model = Model(train_dataset.n_word+1)
    model.train() 
    if gpu:
        model.cuda() 
    model.load_state_dict(torch.load("ckpt/speech.oneway.stft.char.pt"))
    optim = torch.optim.Adam(model.parameters(), 1e-4)
    lossfn = nn.CTCLoss(blank=train_dataset.n_word)
    logsoftmax = nn.LogSoftmax(dim=2)
    n_epoch = 1000
    count = 0
    for b in range(n_epoch):
        loss_collect = []
        for x, d, nx, nd in train_dataloader:
            t, nbatch, c = x.shape 
            h0 = torch.zeros([2, nbatch, model.n_hidden])
            if gpu:
                x = x.cuda() 
                d = d.cuda() 
                nx = nx.cuda() 
                nd = nd.cuda()
                h0 = h0.cuda()
            #print(x.shape, d.shape, nx.shape, nd.shape)
            y = model(x, h0) 
            #print(y.shape)
            m = logsoftmax(y) 
            loss = lossfn(m, d, nx, nd)
            loss.backward() 
            optim.step() 
            optim.zero_grad()
            count += 1
            loss_collect.append(loss.detach().cpu().numpy())
        print(b, count, np.mean(loss_collect))
        torch.save(model.state_dict(), "ckpt/speech.oneway.stft.char.pt")
from scipy.io import wavfile
import scipy.signal as signal 
if __name__ == "__main__":
    main()
    #sr, wav = wavfile.read("data/data_thchs30/A2_0.wav")
    ##wav = torch.from_numpy(wav).float()
    #a, b, f = signal.stft(wav, fs=16000, nperseg=512, noverlap=256)
    ##f = torch.stft(wav, 1024, 512)
    #f = np.abs(f).astype(np.float32)
    #print(f.shape)
    #mfcc = MFCC(sample_rate=16000, n_mfcc=36) 
    #f = mfcc(wav)[0].permute(1, 0)
    #f = pad_sequence([f, f])
    #f = pad_sequence([torch.Tensor([1, 2]), torch.Tensor([1, 2, 3])], batch_first=True)
    #print(f.max(), f.shape)
