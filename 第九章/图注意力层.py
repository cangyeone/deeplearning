import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    GAT中的图注意力层
    """
    def __init__(self, nin, nout):
        super().__init__()
        self.nin = nin
        self.nout = nout
        # 注意力机制层，见第九章
        self.w = nn.Parameter(torch.normal([nin, nout]))
        self.v = nn.Parameter(torch.normal([2*nout, 1]))
        self.leakyrelu = nn.LeakyReLU(0.2) # 斜率为论文中数值

    def forward(self, h, adj):
        """
        h:[节点数N,特征数]
        adj:邻接矩阵[N, N]
        """
        hw = h @ self.w 
        hw1 = hw @ self.v[:self.nout] #[N, 1]
        hw2 = hw @ self.v[self.nout:] #[N, 1] 
        eij = hw1 + hw2.T #[N, N] 
        # 注意力掩码，去除不相邻的矩阵
        mask = -9e15*torch.ones_like(eij)
        attention = torch.where(adj > 0, eij, mask)
        attention = F.softmax(attention, dim=1) # 注意力机制
        hout = attention @ hw 
        return hout 
