from ..tensor import Tensor 
import numpy as np 
def conv2d(inputs: Tensor, filters: Tensor, stride: int, padding="SAME") -> Tensor:
    """卷积计算，后续会补充理论"""
    training = inputs.training or filters.training
    depends_on = []
    B, H, W, C = np.shape(inputs.data)
    K, K, C, C2 = np.shape(filters.data)
    if padding == "SAME":
        H2 = int((H-0.1)//stride + 1)
        W2 = int((W-0.1)//stride + 1)
        pad_h_2 = K + (H2 - 1) * stride - H
        pad_w_2 = K + (W2 - 1) * stride - W
        pad_h_left = int(pad_h_2//2)
        pad_h_right = int(pad_h_2 - pad_h_left)
        pad_w_left = int(pad_w_2//2)
        pad_w_right = int(pad_w_2 - pad_w_left)
        X = np.pad(inputs.data, ((0, 0), 
                            (pad_h_left, pad_h_right),
                            (pad_w_left, pad_w_right), 
                            (0, 0)), 'constant', constant_values=0)
    elif padding == "VALID":
        H2 = int((H - K)//stride + 1)
        W2 = int((W - K)//stride + 1)
        X = inputs
    else:
        raise "parameter error"
    out = np.zeros([B, H2, W2, C2])
    for itr1 in range(B):
        for itr2 in range(H2):
            for itr3 in range(W2):
                for itrc in range(C2):
                    itrh = itr2 * stride
                    itrw = itr3 * stride
                    out[itr1, itr2, itr3, itrc] = np.sum(X[itr1, itrh:itrh+K, itrw:itrw+K, :] * filters.data[:,:,:,itrc]) 
    if inputs.training:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            """梯度计算"""
            error = np.zeros_like(X)
            for itr1 in range(B):
                for itr2 in range(H2):
                    for itr3 in range(W2):
                        for itrc in range(C2):
                            itrh = itr2 * stride
                            itrw = itr3 * stride
                            error[itr1, itrh:itrh+K, itrw:itrw+K, :] += grad[itr1, itr2, itr3, itrc] *  filters[:,:,:,itrc]
            return error
        depends_on.append((inputs, grad_fn1))

    if filters.training:
        """梯度计算"""
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            dw = np.zeros_like(filters.data)
            for itr1 in range(B):
                for itr2 in range(H2):
                    for itr3 in range(W2):
                        for itrc in range(C2):
                            itrh = itr2 * stride
                            itrw = itr3 * stride
                            dw[:, :, :, itrc] += grad[itr1, itr2, itr3, itrc] * X[itr1, itrh:itrh+K, itrw:itrw+K, :]
            return dw
        depends_on.append((filters, grad_fn2))
    return Tensor(out, training, depends_on, "conv2d")  



def conv2d_orig(x, w, stride=1, padding=0):
    """
    卷积神经网络的原始实现
    内存需求较少，但是内存不连续计算较慢
    """
    x = np.pad(x, [(0, 0), (0, 0), (padding, padding), (padding, padding)])
    b, c1, h1, w1 = x.shape 
    c2, c1, k1, k2 = w.shape 
    # 输出图像大小
    h2, w2 = (h1-k1+1)//stride, (w1-k2+1)//stride
    out = np.zeros([b, c2, h2, w2]) 
    for ib in range(b):
        for ic2 in range(c2):
            for ih2 in range(h2):
                for iw2 in range(w2):
                    ih = ih2 * stride
                    iw = iw2 * stride
                    out[ib, ic2, ih2, iw2] = \
                        np.sum(
                            x[ib, :, ih:ih+k1, iw:iw+k2] * \
                                w[ic2]
                            )
    return out 

def relu(t1: Tensor) -> Tensor:
    """通过激活函数"""
    training = t1.training
    depends_on = []
    if t1.training:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """梯度计算"""
            grad2 = np.copy(grad) 
            grad2 = grad * (t1.data>0).astype(np.float64)
            return grad2
        depends_on.append((t1, grad_fn))
    return Tensor(np.clip(t1.data, 0, np.inf), training, depends_on, "conv2d") 
def concat(array_list, axis=-1) -> Tensor:
    """矩阵连接操作"""
    depends_on = [] 
    training = False 
    datas = np.concatenate([itr.data for itr in array_list], axis) 
    dims = [itr.data.shape[axis] for itr in array_list]
    for itr in array_list:
        if itr.training:
            training = itr.training
    for idx, itr in enumerate(array_list):
        if itr.training:
            def grad_fn(grad: np.ndarray) -> np.ndarray:
                return np.split(grad, dims, axis)[idx] 
            depends_on.append((itr, grad_fn))
    return Tensor(datas, training, depends_on, "concat") 

def cross_entropy(y, d):
    N = len(y)
    e = np.exp(np.clip(y.data, -100, 100))
    e = e/np.sum(e, axis=1, keepdims=True) 
    loss = np.sum(-np.log(e[np.arange(N), d.data]))/N 
    training = y.training
    if training:
        def grad_fn(grad): 
            """参考交叉熵"""
            grad = e.copy() 
            grad[np.arange(N), d.data] -= 1
            return grad / N 
        depends_on = [(y, grad_fn)]
    else:
        depends_on = []
    return Tensor(loss, training, depends_on)


def conv2dbase(x, w, stride):
    """卷积的正向计算"""
    b, c1, h1, w1 = x.shape
    c2, c1, k1, k2 = w.shape 
    h2, w2 = (h1-k1)//stride+1, (w1-k2)//stride+1
    # 卷积核心对应位置索引
    idxw = np.arange(k2)[np.newaxis, :] + np.zeros([c1, k1, 1]) 
    idxh = np.arange(k1)[:, np.newaxis] + np.zeros([c1, 1, k2]) 
    idxc = np.arange(c1)[:, np.newaxis, np.newaxis] + np.zeros([1, k1, k2])
    idxw = idxw.reshape([1, -1]) + \
        (np.arange(w2) * stride + np.zeros([h2, 1])).reshape([-1, 1])
    idxh = idxh.reshape([1, -1]) + \
        (np.arange(h2)[:, np.newaxis] * stride+np.zeros([w2])).reshape([-1, 1])
    idxc = idxc.reshape([1, -1]) + np.zeros([h2, w2]).reshape([-1, 1]) 
    idxw = idxw.astype(np.int32) 
    idxh = idxh.astype(np.int32) 
    idxc = idxc.astype(np.int32) 
    w = w.reshape([c2, c1*k1*k2]).T 
    col = x[:, idxc, idxh, idxw]
    cv = col @ w # 矩阵求导章节，回顾矩阵求导章节
    reim = cv.reshape([b, h2, w2, c2])
    reim = reim.transpose([0, 3, 1, 2])
    return reim, col.reshape([-1, c1*k1*k2])

def conv2d(inputs, weight, stride=1, pad=0):
    """
    卷积函数包括反向传播过程
    """
    x = inputs.data 
    w = weight.data 
    b, c1, h1, w1 = inputs.shape 
    xp = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)]) 
    y, col = conv2dbase(xp, w, stride) 
    c2, c1, k1, k2 = w.shape 
    training = inputs.training or weight.training
    depends_on = []
    if inputs.training:
        def grad_fn(e):
            c2, c2, h2, w2 = e.shape 
            # 反向计算过程中由于步长需要重复多次
            e = np.repeat(e, stride, axis=2)
            e = np.repeat(e, stride, axis=3)
            hidx = np.arange(h2) * stride
            widx = np.arange(w2) * stride  
            # 多出来的部分补0
            for s in range(stride-1):
                e[:, :, hidx+s+1, :] = 0 
                e[:, :, :, widx+s+1] = 0 
            # 对周围需要补0 
            ep = np.pad(e, [(0, 0), (0, 0), (k1-1, k1-1), (k2-1, k2-1)]) 
            # 卷积核心翻转180度
            rw = w[:, :, ::-1, ::-1] 
            rw = rw.transpose([1, 0, 2, 3]) 
            # e关于输入x的导数
            dx, colt = conv2dbase(ep, rw, 1) 
            # 去除补0的位置
            dx = dx[:, :, pad:h1+pad, pad:w1+pad]
            return dx
        depends_on.append((inputs, grad_fn))
    if weight.training:
        def grad_fn(e):
            # 误差转为矩阵
            ecol = e.transpose([0, 2, 3, 1]).reshape([-1, c2])
            # 计算e关于w的导数即矩阵求导 
            dw = ecol.T @ col # 参考矩阵求导部分，求可训练参数导数
            dw = dw.reshape([c2, c1, k1, k2])
            return dw    
        depends_on.append((weight, grad_fn))     
    return Tensor(y, training, depends_on)

def concat(array_list, axis=-1) -> Tensor:
    """矩阵连接操作"""
    depends_on = [] 
    training = False 
    datas = np.concatenate([itr.data for itr in array_list], axis) 
    dims = [itr.data.shape[axis] for itr in array_list]
    for itr in array_list:
        if itr.training:
            training = itr.training
    for idx, itr in enumerate(array_list):
        if itr.training:
            def grad_fn(grad: np.ndarray) -> np.ndarray:
                return np.split(grad, dims, axis)[idx] 
            depends_on.append((itr, grad_fn))
    return Tensor(datas, training, depends_on, "concat") 

