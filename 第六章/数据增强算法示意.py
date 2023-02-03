from plotconfig import * 
import torch 
import torchvision.transforms as transforms
# 随机仿射变换，改变物体角度、大小、维度
trans1 = transforms.RandomAffine((-30, 30), scale=(0.5, 1.0))
# 随机剪裁
trans2 = transforms.RandomCrop((100, 100))
# 随机擦除
trans3 = transforms.RandomErasing()
# 随机调整亮度、饱和度等，参考书中图形章节
trans4 = transforms.ColorJitter()
# 随机调整不同分辨率
trans5 = transforms.RandomResizedCrop((100, 100))
# 随机调整对比度
trans6 = transforms.RandomAutocontrast()



fig = plt.figure(1, figsize=(12, 12), dpi=100) 
gs = grid.GridSpec(6, 6)
img = plt.imread("data/img3.jpg")[::5, ::5, :]
x = torch.tensor(img, dtype=torch.float32)/255 
x = x.permute(2, 1, 0).unsqueeze(0) 
names = ["仿射变换", "剪裁", "擦除", "亮度、饱和度", "分辨率", "对比度"]
trans = [trans1, trans2, trans3, trans4, trans5, trans6]
for i in range(6):
    print(names[i])
    ts = trans[i]
    for j in range(6):
        y = ts(x) 
        y = y.squeeze().permute(2, 1, 0) 
        y = y.cpu().numpy() 
        if j == 0:
            ax = fig.add_subplot(gs[i, j]) 
            if i == 0:
                print(i, j, names[i])
                ax.set_title("原始图像")
            ax.imshow(img)
            ax.set_xticks(()) 
            ax.set_yticks(()) 
            ax.set_ylabel(names[i])
        else:
            ax = fig.add_subplot(gs[i, j]) 
            if i == 0:
                ax.set_title("随机变换")
            ax.imshow(y)
            ax.set_xticks(()) 
            ax.set_yticks(()) 
           

plt.savefig("导出图像/图像增强.jpg")
plt.savefig("导出图像/图像增强.svg")