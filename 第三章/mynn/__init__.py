from .tensor import Tensor 
from .optim import optim 
import pickle 

def load(file_name):
    with open(file_name, "rb") as f:
        fdict = pickle.load(f)
    return fdict 

def save(fdict, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(fdict, f)