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
    if "科技" not in line:continue 
    text = line.split("\t")[1] 
    segt = jieba.lcut(text) 
    datas.extend(segt) 

words = " ".join(datas)



# 运用matplotlib展现结果
fig = plt.figure(1, figsize=(12, 12), dpi=150)
gs = grid.GridSpec(2, 1, figure=fig)
ax = fig.add_subplot(gs[0, 0])
word_cloud = WordCloud(font_path="simsun.ttc", width=2400, height=1200,  # 设置词云字体
                       background_color="white") # 去掉的停词
word_cloud.generate(words)
ax.imshow(word_cloud)
set_axis(ax) 
ax.set_title("a)", x=0, y=1, va="bottom", ha="left")

stopfile = open("data/stop_words_bd.txt", "r", encoding="utf-8") 
stopwords = stopfile.read().split("\n")
stopfile.close()
ax = fig.add_subplot(gs[1, 0])
word_cloud = WordCloud(font_path="simsun.ttc", width=2400, height=1200,  # 设置词云字体
                       background_color="white", 
                       stopwords=stopwords) # 去掉的停词
word_cloud.generate(words)
ax.imshow(word_cloud)
set_axis(ax) 
ax.set_title("b)", x=0, y=1, va="bottom", ha="left")


plt.savefig("导出图像/词云图.png")
plt.savefig("导出图像/词云图.svg")