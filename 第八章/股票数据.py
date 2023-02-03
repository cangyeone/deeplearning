from plotconfig import * 
import tushare as ts 
import datetime 

number = '600848'
stock = ts.get_hist_data(number) #一次性获取全部日k线数据 
data = stock.values 
data = data[::-1]
fig = plt.figure(1, figsize=(16, 9), dpi=100) 
gs = grid.GridSpec(1, 2) 
ax = fig.add_subplot(gs[0]) 
ax.plot(data[:, 0], c="r", alpha=0.5, label="开盘价")
ax.plot(data[:, 1], c="g", alpha=0.5, label="最高价")
ax.plot(data[:, 2], c="b", alpha=0.5, label="收盘价")
ax.grid(True)
ax.set_title(f"a)", x=0, y=1, ha="left", va="bottom")
ax.set_xlabel("日期")
ax.set_ylabel("数值")
ax.legend(loc="upper right")

ax = fig.add_subplot(gs[1]) 
ax.plot(data[1:, 0]-data[:-1, 0], c="r", alpha=0.5, label="开盘价")
ax.plot(data[1:, 1]-data[:-1, 1], c="g", alpha=0.5, label="最高价")
ax.plot(data[1:, 2]-data[:-1, 2], c="b", alpha=0.5, label="收盘价")
ax.grid(True)
ax.set_title(f"b)", x=0, y=1, ha="left", va="bottom")
ax.set_xlabel("日期")
ax.set_ylabel("数值")
ax.legend(loc="upper right")
plt.savefig("导出图像/股票数据1.jpg")
plt.savefig("导出图像/股票数据1.svg")


