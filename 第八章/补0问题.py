import torch 
from torch.nn.utils.rnn import pad_sequence

w1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
w2 = torch.tensor([1, 2, 3, 4], dtype=torch.long) 
w3 = torch.tensor([1, 2, 3], dtype=torch.long) 
x_paded = pad_sequence([w1, w2, w3], padding_value=0) #Shape:[5, 3]


from torch.nn.utils.rnn import pack_sequence 
w1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
w2 = torch.tensor([1, 2, 3, 4], dtype=torch.long) 
w3 = torch.tensor([1, 2, 3], dtype=torch.long) 
# 如果不是长度从大到小排列，需要设置enforce_sorted
# 方法会自动从小到大进行排列
x_packed = pack_sequence([w3, w2, w1], enforce_sorted=False) 


from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
# 返回补零数据，并返回每个样本实际长度
x_unpacked, x_length = pad_packed_sequence(x_packed, padding_value=0) 
# 返回打包数据，需要给定实际样本长度
x_packed = pack_padded_sequence(x_paded, lengths=x_length, enforce_sorted=False)

print(x_unpacked.shape, x_length, x_packed)