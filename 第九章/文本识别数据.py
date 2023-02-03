from plotconfig import * 
import os 

basedir = "data/text/reco"
text_path = os.path.join(basedir, "gt1.txt")
texts = [] 
paths = []
with open(text_path, "r", encoding="utf-8") as f:
    for line in f.readlines():
        sline = line.strip().split(", ")
        lab = sline[1].replace('"', "") 
        pth = sline[0]
        texts.append(lab)
        paths.append(os.path.join(basedir, pth)) 



fig = plt.figure(1, figsize=(12, 9), dpi=100) 
gs = grid.GridSpec(5, 5) 
for i in range(5):
    for j in range(5):
        idx = i * 10 + j 
        img = plt.imread(paths[idx]) 
        lab = texts[idx]
        ax = fig.add_subplot(gs[i, j]) 
        ax.imshow(img) 
        ax.set_xticks(()) 
        ax.set_yticks(()) 
        ax.set_xlabel(lab) 
plt.savefig("导出图像/文本识别数据.svg")
plt.savefig("导出图像/文本识别数据.jpg")
