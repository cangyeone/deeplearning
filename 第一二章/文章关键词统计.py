# 需要安装wordcloud
# 使用WordCloud生成词云
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer 

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['font.size'] = "16"  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def set_axis(ax):
    ax.set_xticks(())
    ax.set_yticks(())
file_ = open("data/cnews.test.txt", "r", encoding="utf-8")
datas = [] 
for line in file_.readlines():
    name, text = line.split("\t") 
    if "科技" not in name:continue 
    segt = jieba.lcut(text) 
    datas.append(" ".join(segt)) 


tool = TfidfVectorizer() 
data = tool.fit_transform(datas)

aidx = 100
row = data.getrow(aidx).todense()
row = np.copy(row)[0]
sidx = np.argsort(row)[::-1] 
row = row[sidx]
words = []
id2word = tool.get_feature_names()
print(datas[aidx].replace(" ", ""))
for i in sidx[:20]:
    words.append(id2word[i])
print(" ".join(words))
