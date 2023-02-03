from ..tensor import Tensor 
import numpy as np 
def tcat(tensors, dim=0) -> Tensor:
    """获取索引，切片操作"""
    datas = [t.data for t in tensors]
    training = tensors[0].training
    for t in tensors:
        training = training or t.training 
    data = np.concatenate(datas, axis=dim)
    nsplit = [] 
    s = 0 
    for t in datas:
        s += t.shape[dim] 
        nsplit.append(s)
    if training:
        depends_on = []
        for i in range(len(datas)):
            def grad_fn(grad: np.ndarray) -> np.ndarray:
                """梯度计算就是切片位置"""
                grads = np.split(grad, nsplit, axis=dim)
                return grads[i]
            depends_on.append((t, grad_fn))
    else:
        depends_on = []
    return Tensor(data, training, depends_on, "slice")
def cat(array_list, dim=-1) -> Tensor:
    """矩阵连接操作"""
    depends_on = [] 
    training = False 
    # 对数据进行连接
    datas = np.concatenate([itr.data for itr in array_list], dim) 
    dims = [] 
    s = 0 
    for t in array_list:
        s += t.shape[dim] 
        dims.append(s)
    for itr in array_list:# 列表中如果有一个可训练，连接结果便可以训练
        if itr.training:
            training = itr.training
    for idx, itr in enumerate(array_list):
        if itr.training:
            def grad_fn(grad: np.ndarray) -> np.ndarray:
                # 对反向传播梯度进行分割
                grads = np.split(grad, dims, dim)
                return grads[idx]
            depends_on.append((itr, grad_fn))
    return Tensor(datas, training, depends_on, "cat") 
def split(t1, num, dim=-1):
    """矩阵分割"""
    depends_on = [] 
    training = t1.training  
    # 对数据进行分割
    datas = np.split(t1.data, num, axis=dim)
    tensors = []
    if training:
        for idx, dt in enumerate(datas):
            def grad_fn(grad: np.ndarray) -> np.ndarray:
                # 将梯度进行连接
                grads = [np.zeros_like(dt) for dt in datas]
                grads[idx] = grad 
                grads = np.concatenate(grads, axis=dim)
                return grads
            depends_on.append((t1, grad_fn))
            ten = Tensor(dt, training, depends_on, "split") 
            tensors.append(ten)
    return tensors 


def zeros(shape, training=False):
    return Tensor(np.zeros(shape), training, [], "zeros")


def tanh(t1: Tensor) -> Tensor:
    """通过激活函数"""
    training = t1.training
    depends_on = []
    data = np.tanh(t1.data)
    if t1.training:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """梯度计算"""
            grad2 = np.copy(grad) 
            grad2 = grad * (1 - np.tanh(t1.data) ** 2)
            return grad2
        depends_on.append((t1, grad_fn))
    return Tensor(data, training, depends_on, "tanh") 

def sigmoid(t1: Tensor) -> Tensor:
    """通过激活函数"""
    training = t1.training
    depends_on = []
    data = 1/(1+np.exp(t1.data))
    if t1.training:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """梯度计算"""
            grad2 = np.copy(grad) 
            grad2 = grad * (data * (1-data))
            return grad2
        depends_on.append((t1, grad_fn))
    return Tensor(data, training, depends_on, "tanh") 
