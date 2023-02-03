from unicodedata import bidirectional
import torch.nn as nn 
import torch 


class Stock(nn.Module):
    """
    股票预测改进模型
    """
    def __init__(self):
        super().__init__() 
        n_class = 10 # 公司类别数量
        # 星期编码
        self.days_emb = nn.Embedding(7, 32) 
        # 类别编码 
        self.class_emb = nn.Embedding(n_class, 32)
        # 将三个特征转换为32个特征，相当于Embedding
        self.input_layer = nn.Linear(3, 32)
        # 循环神经网络是单向模型
        self.rnn = nn.GRU(32, 32, 2, bidirectional=False) 
        # 输出层，依然是三个
        self.output_layer = nn.Linear(32, 3) 
    def forward(self, x, days, class_): 
        """
        x:股票数据值（float）[天数,样本数量,3个值]->[T,B,C] 
        days:星期（long）[T,B]
        class_:股票公司类别（long）[B]
        """
        T, B, C = x.shape 
        x1 = self.input_layer(x) # 股票值编码
        x2 = self.days_emb(days) # 星期编码
        x3 = self.class_emb(class_) # 类别编码
        x3 = x3.unsqueeze(dim=0) # 所有时间步加入相同类别
        print(x1.shape, x2.shape, x3.shape)
        x = x1 + x2 + x3 
        # 初始状态为0
        h = torch.zeros([self.rnn.num_layers, B, 32], device=x.device) 
        y, h = self.rnn(x, h) 
        y = self.output_layer(y) 
        return y 
model = Stock()
x = torch.zeros([10, 5, 3]) 
d = torch.zeros([10, 5]).long() 
c = torch.zeros([5]).long()

y = model(x, d, c) 
print(y.shape)
