import numpy as np 
from plotconfig import * 

file_ = np.load("data/mnist.npz")
x_train = file_["x_train"]
d_train = file_["y_train"]
x_test = file_["x_test"]
d_test = file_["y_test"]
x_train = x_train.astype(np.float32)/255 
x_test = x_test.astype(np.float32)/255 

fig = plt.figure(1, figsize=(12, 9), dpi=100)
gs = grid.GridSpec(3, 4)
for i in range(3):
    for j in range(4):
        x = x_train[i*4+j]
        d = d_train[i*4+j]
        ax = fig.add_subplot(gs[i, j])
        ax.matshow(x, cmap=plt.get_cmap("Greys"))
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_xlabel(f"类别:{d}")
plt.savefig("导出图像/手写数字.jpg")
plt.savefig("导出图像/手写数字.pdf")
print(x_train.shape)
          