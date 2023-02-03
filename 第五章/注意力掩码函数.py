import torch 
def generate_square_subsequent_mask(sz, device):
    # 注意力掩码为三角矩阵
    mask = (torch.triu(torch.ones(
        (sz, sz), device=device)) == 1).transpose(0, 1)
    # Attention为0，即不能包含后文信息，应当添加-inf
    mask = mask.float().masked_fill(
        mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
    return mask