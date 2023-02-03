from sklearn.datasets import fetch_20newsgroups
outfile = open("data/en.news.txt", "w", encoding="utf8")
news = fetch_20newsgroups(data_home="data")
for txt in news.data:
    outfile.write(txt)