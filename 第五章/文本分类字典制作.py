file_train = open("data/cnews.train.txt", "r", encoding="utf-8") 
train_text = file_train.read()

all_words = set(train_text) # 获取不重复的词
# 给每个词一个编号
word2id = dict(zip(all_words, range(len(all_words))))
with open("ckpt/word2id", "w", encoding="utf-8") as f:
    f.write(str(word2id))


label2id = {}
nd = 0 
for line in train_text.split("\n")[:-1]:
    d, x = line.split("\t") #tab作为标签间隔
    if d not in label2id:
        label2id[d] = nd 
        nd += 1
with open("ckpt/label2id", "w", encoding="utf-8") as f:
    f.write(str(label2id))