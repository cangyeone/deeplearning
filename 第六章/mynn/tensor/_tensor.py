"""
自动求导功能的实现
参考github:autograd
"""

import numpy as np

def data_trans(data):
    """转换为ndarray类型数据"""
    if isinstance(data, np.ndarray):return data
    else:return np.array(data)
def tensor_trans(data):
    """转换为Tensor类型"""
    if isinstance(data, Tensor):return data
    else:return Tensor(data)

class Tensor():
    def __init__(self, data, training=False, depends_on=[], name="input"):
        self._data = data_trans(data)# 转换数据为ndarray
        self.training = training# 定义是否可训练
        self.shape = self._data.shape# 模型的shape
        self.grad = None # 梯度
        self.depends_on = depends_on # 数据的依赖的变量 
        self.step = -1      
        self.name = name             # 当前节点的名称
        if self.training:
            self.zero_grad()         # 如果可训练的话计算梯度

    def zero_grad(self):
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64), training=False)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray):
        self._data = new_data

    def __len__(self):
        return len(self._data)
    def __repr__(self):
        return f"Tensor:({self._data}, training={self.training})"

    def __add__(self, other):
        """加法"""
        return _add(self, tensor_trans(other))

    def __radd__(self, other):
        """右加"""
        return _add(tensor_trans(other), self)

    def __mul__(self, other):
        """乘法"""
        return _mul(self, tensor_trans(other))

    def __rmul__(self, other):
        """右乘"""
        return _mul(tensor_trans(other), self)

    def __matmul__(self, other):
        """矩阵点乘，运算符号@"""
        return _matmul(self, tensor_trans(other))
    def __rmatmul__(self, other):
        """矩阵点乘，运算符号@"""
        return _matmul(tensor_trans(other), self)

    def __sub__(self, other):
        """减法"""
        return _sub(self, tensor_trans(other))

    def __rsub__(self, other):
        """右减"""
        return _sub(tensor_trans(other), self)

    def __neg__(self):
        """取反"""
        return _neg(self)

    def __getitem__(self, idxs):
        """矩阵获取索引"""
        return _getitem(self, idxs)

    def backward(self, grad=None):
        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0) 
            else:
                grad = Tensor(np.ones(self.shape))
        self.grad.data = self.grad.data + grad.data  
        for temp in self.depends_on:
            tensor, grad_fn = temp 
            #print("梯度", self.name, np.abs(grad.data).sum(), grad.data.mean())
            backward_grad = grad_fn(grad.data)
            tensor.backward(Tensor(backward_grad))
    def sum(self):
        return tensor_sum(self)
    def mean(self):
        return tensor_mean(self)
    def mm(self, data):
        return _matmul(self, tensor_trans(data)) 
    def clamp(self, min=0, max=np.inf):
        return _clip(self, min, max)
    def __pow__(self, n):
        return _pow(self, n)
    def pow(self, n):
        return _pow(self, n)
    def reshape(self, idx):
        return _reshape(self, idx)
    def sigmoid(self):
        return sigmoid(self) 
    def tanh(self):
        return tanh(self)

def tensor_sum(t: Tensor) -> Tensor:
    """
    将所有元素进行相加
    """
    data = t.data.sum()
    training = t.training

    if training:
        def grad_fn(grad): 
            """
            本层梯度就是本身，前一层传播的梯度乘以1.
            """
            return grad * np.ones_like(t.data)
        depends_on = [(t, grad_fn)]

    else:
        depends_on = []

    return Tensor(data,
                  training,
                  depends_on, "sum")
def tensor_mean(t: Tensor) -> Tensor:
    """
    将所有元素进行相加
    """
    data = t.data.mean()
    training = t.training
    if training:
        def grad_fn(grad): 
            """
            本层梯度就是本身，前一层传播的梯度乘以1.
            """
            return grad * np.ones_like(t.data) / len(t.data)
        depends_on = [(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data,
                  training,
                  depends_on, "sum")
def _add(t1: Tensor, t2: Tensor) -> Tensor:
    """加法"""
    data = t1.data + t2.data
    training = t1.training or t2.training
    depends_on = []
    if t1.training:
        def grad_fn1(grad):
            # 在梯度计算假设梯度维度为[N, H, W, C] 而加入的偏置b维度为[C] 
            # 所以实际上应当对于 N, H, W 维度进行相加才是b的偏导数。
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            # 在维度为1的地方进行相加
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append((t1, grad_fn1))

    if t2.training:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # 在梯度计算假设梯度维度为[N, H, W, C] 而加入的偏置b维度为[C] 
            # 所以实际上应当对于 N, H, W 维度进行相加才是b的偏导数。
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            # 在维度为1的地方进行相加
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        depends_on.append((t2, grad_fn2))

    return Tensor(data,
                  training,
                  depends_on, 
                  "add")

def _mul(t1: Tensor, t2: Tensor) -> Tensor:
    """乘法"""
    data = t1.data * t2.data
    training = t1.training or t2.training
    depends_on = []
    if t1.training:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad * t2.data
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            # 在维度为1的地方需要相加
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        depends_on.append((t1, grad_fn1))

    if t2.training:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = grad * t1.data

            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            # 在维度为1的地方需要相加
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        depends_on.append((t2, grad_fn2))
    return Tensor(data,               
                  training,      
                  depends_on, "mul")         

def _neg(t: Tensor) -> Tensor:
    """取反"""
    data = -t.data
    training = t.training
    if training:
        depends_on = [(t, lambda x: -x)]
    else:
        depends_on = []

    return Tensor(data, training, depends_on, "neg")
def _pow(t: Tensor, n) -> Tensor:
    data = t.data ** n 
    depends_on = []
    if t.training:
        def grad_fn(grad) -> np.ndarray: 
            return grad * n * t.data ** (n-1) 
        depends_on.append((t, grad_fn))
    return Tensor(data, t.training, depends_on, "pow")
def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    """减法"""
    return t1 + -t2

def _matmul(t1: Tensor, t2: Tensor) -> Tensor:
    """
    矩阵点乘，此部分内容可以参阅矩阵求导部分
    计算为y = t1 @ t2 
    """
    data = t1.data @ t2.data 
    training = t1.training or t2.training # 新节点是否需要计算梯度
    depends_on = [] # 新节点的依赖变量
    if t1.training:#如果可训练才进行求导
        def grad_fn1(grad: np.ndarray):
            """
            grad为上一步反向传播的梯度grad=∂L/∂y
            梯度计算，求∂L/∂t1=grad·(∂y/∂t1)=grad·t2.T
            参考矩阵求导部分
            """
            return grad @ t2.data.T
        depends_on.append((t1, grad_fn1)) 

    if t2.training:#如果可训练才进行求导
        def grad_fn2(grad: np.ndarray):
            """
            grad为上一步反向传播的梯度grad=∂L/∂y
            梯度计算，求∂L/∂t2=grad·(∂y/∂t2)=t1.T·grad
            参考矩阵求导部分
            """
            return t1.data.T @ grad
        depends_on.append((t2, grad_fn2))
    return Tensor(data,# 数据
                  training,#是否可训练
                  depends_on,#依赖（变量和导数） 
                  "matmul")
def _clip(t: Tensor, smin, smax) -> Tensor: 
    depends_on = []
    data = t.data 
    if t.training:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            retgrad = np.copy(grad) 
            retgrad = retgrad.reshape(-1)
            retgrad[data.reshape(-1)<smin] = 0 
            retgrad[data.reshape(-1)>smax] = 0 
            return np.reshape(retgrad, grad.shape) 
        depends_on.append((t, grad_fn))
    data = np.clip(t.data, smin, smax)
    return Tensor(data, t.training, depends_on, "clip")
def _getitem(t: Tensor, idxs) -> Tensor:
    """获取索引，切片操作"""
    data = t.data[idxs]
    training = t.training
    if training:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """梯度计算就是切片位置"""
            bigger_grad = np.zeros_like(t.data)
            bigger_grad[idxs] = grad
            return bigger_grad
        depends_on = [(t, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, training, depends_on, "slice")


def _reshape(t: Tensor, idxs) -> Tensor:
    """reshape操作"""
    ishape = t.data.shape 
    data = t.data.reshape(idxs)
    training = t.training

    if training:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """梯度计算就是切片位置"""
            bigger_grad = grad.reshape(ishape)
            return bigger_grad
        depends_on = [(t, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, training, depends_on, "reshape")


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