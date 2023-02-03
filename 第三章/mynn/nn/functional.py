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