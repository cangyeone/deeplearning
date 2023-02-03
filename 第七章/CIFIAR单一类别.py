from plotconfig import * 
import pickle 
import numpy as np 
with open("data/cifar-10-batches-py/data_batch_1", 'rb') as fo:
    data_dict = pickle.load(fo, encoding="bytes")
labels = data_dict[b"labels"]
images = data_dict[b"data"]
images = np.array(images) 
labels = np.array(labels) 
images = images[labels==0].astype(np.float32)/255 
images[:, :] = np.mean(images, axis=0)
fig = plt.figure(1, figsize=(18, 9), dpi=100) 
gs = grid.GridSpec(2, 4) 
names = ["飞机", "手机", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
for i in range(2):
    for j in range(4):
        ax = fig.add_subplot(gs[i, j]) 
        idx = i * 8 + j 
        ax.imshow(np.reshape(images[idx], [3, 32, 32]).transpose(1, 2, 0)) 
        ax.set_xticks(()) 
        ax.set_yticks(())
        x1, x2, y1, y2 = 10, 11, 10, 11
        ax.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], c="r")
        #ax.set_xlabel(f"{names[labels[idx]]}")
        #ax.set_xlabel(f"${names[labels[idx]]}$")

plt.savefig("导出图像/cifiar.singleclass.mean.svg")
plt.savefig("导出图像/cifiar.singleclass.mean.jpg")