import torch.nn as nn
import torch
import torch.nn.functional as F 

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        ConvLSTM模型结构
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        # 传统卷积 
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,#LSTM卷积特征数量为4倍
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, x, state):
        h, c = state

        combined = torch.cat([x, h], dim=1)  # 在特征维度上进行连接

        out = self.conv(combined)
        # 将其分为3个门控结果+1个输出结果
        g1, g2, g3, o = torch.split(out, self.hidden_dim, dim=1)
        # 使用不同激活函数
        g1 = g1.sigmoid() # 遗忘门
        g2 = g2.sigmoid() # 更新门
        g3 = g3.sigmoid() # 输出门 
        g3 = torch.sigmoid(g3)
        o = torch.tanh(o)
        c_next = g1 * c + g2 * o 
        h_next = g3 * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            #print("NumLayer", layer_idx)
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class ConvBNReLU(nn.Module):
    def __init__(self, nin, nout, ks, stride) -> None:
        super().__init__()
        pad = (ks-1)//2
        self.layers = nn.Sequential(
            nn.Conv2d(nin, nout, ks, stride, padding=pad), 
            nn.BatchNorm2d(nout), 
            nn.ReLU()
        )
    def forward(self, x):
        y = self.layers(x) 
        return y 
class ConvBNReLUTrans(nn.Module):
    def __init__(self, nin, nout, ks, stride) -> None:
        super().__init__()
        pad = (ks-1)//2
        self.layers = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=stride), 
            nn.Conv2d(nin, nout, ks, 1, padding=pad), 
            nn.BatchNorm2d(nout), 
            nn.ReLU()
        )
    def forward(self, x):
        y = self.layers(x) 
        return y 
class InvNet(nn.Module):
    def __init__(self):
        super(InvNet, self).__init__() 
        self.encoder = nn.Sequential(
            ConvBNReLU(3, 8, 3, 2), #2
            ConvBNReLU(8, 8, 3, 2), #4 
            ConvBNReLU(8, 16, 3, 2), #8 
            ConvBNReLU(16, 16, 3, 2), #16 
            ConvBNReLU(16, 32, 3, 2), #32 
            ConvBNReLU(32, 32, 3, 2), #64
            ConvBNReLU(32, 64, 3, 2), #128
            ConvBNReLU(64, 64, 3, 2), #256 
            ConvBNReLU(64, 128, 3, 1), #512 
        )
        self.clstm = ConvLSTM(128, [128]*2, [(3, 3)]*2, 2) 
        self.decoder = nn.Sequential(
            ConvBNReLUTrans(128, 64, 3, 2), #2
            ConvBNReLUTrans(64, 64, 3, 2), #4
            ConvBNReLUTrans(64, 32, 3, 2), #8
            ConvBNReLUTrans(32, 32, 3, 2), #16
            ConvBNReLUTrans(32, 16, 3, 2), #32
            ConvBNReLUTrans(16, 16, 3, 2), #64
            ConvBNReLUTrans(16, 8, 3, 2), #128
            ConvBNReLUTrans(8, 8, 3, 2), #256
        )
        self.output = nn.Conv2d(8, 1, 3, padding=1)
    def forward(self, x):
        B, T, H, W, C = x.shape 
        S = 256
        x = x.permute(1, 0, 4, 2, 3)
        x = x.reshape([T * B, C, H, W])
        x = self.encoder(x) 
        x = torch.reshape(x, [T, B, 128, H//S, W//S])
        x, s = self.clstm(x)
        x = x[-1][:, -1, :, :, :]
        x = self.decoder(x) 
        x = self.output(x) 
        x = x.permute(0, 2, 3, 1)
        x = x.sigmoid() * 2.5 + 2.5 
        return x 

from utils.data import DataTrain
import numpy as np 
import time 

import time
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec 
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = (16, 16)
plt.rcParams['figure.dpi'] = 150
plt.rcParams["font.size"] = 16
def main():
    device = torch.device("cuda:0")
    model_name = "ckpt/directly-inv-4.pth"
    model = InvNet() 
    model.train() 
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0e-3)
    data_tool = DataTrain() 
    ls = []
    t1 = time.perf_counter()
    acct = 0
    try:
        #pass 
        model.load_state_dict(torch.load(model_name))
    except:
        pass  
    ofile = open("logdir/loss.directly.inv4.txt", "a")
    for step in range(100000):
        X, D = data_tool.next_batch(batch_size=32)
        x1 = torch.tensor(X.astype(np.float32), device=device) 
        d1 = torch.tensor(D.astype(np.float32), device=device) 
        tt1 = time.perf_counter()
        y1 = model(x1) 
        loss = (torch.square((y1-d1))).sum() #MSE作为约束
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 
        tt2 = time.perf_counter()
        acct += tt2 - tt1 
        ls.append(loss.cpu().detach())
        if step % 5 ==0:
            torch.save(model.state_dict(), model_name)
if __name__ == "__main__":
    main()