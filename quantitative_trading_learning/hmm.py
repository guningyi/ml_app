#!/usr/bin/python
#-*-coding:utf-8 -*-

from hmmlearn.hmm import GaussianHMM
import numpy as np
from matplotlib import cm, pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import datetime


if __name__ == '__main__':

    beginDate = '2010/01/04'
    endDate = '2016/06/02'
    # 6个隐藏状态
    n = 6
    data = np.loadtxt('C:\\Git\\ml_basic\\26.HMM\\26.SH600000.txt',
                      dtype={
                             'names': ('closingPx', 'TotalVolumeTraded', 'TotalTurnover', 'HighPx', 'LowPx', 'Date'),
                             'formats': (np.float, np.float, np.float, np.float, np.float, 'S10')
                      },
                      delimiter='\t',
                      skiprows=2,
                      usecols=(4, 5, 6, 2, 3, 0))
    #data[0:4]
    # 0收盘价(ClosingPx) 1当日总成交量(TotalVolumeTraded or volumn) 2当日总成交额(TotalTurnover or amount) 3最高价(HighPx) 4最低价(LowPx)
    #print data

    volume = data['TotalVolumeTraded'] #每日成交量
    close = data['closingPx']  #每日收盘价

    # 计算每日最高最低价格的对数差值，作为特征状态的一个指标。
    logDel = np.log(np.array(data['HighPx'])) - np.log(np.array(data['LowPx']))
    #print logDel

    # 这个作为后面计算收益使用
    logRet_1 = np.array(np.diff(np.log(close)))

    # 计算每5日的指数对数收益差，作为特征状态的一个指标。
    logRet_5 = np.log(np.array(close[5:])) - np.log(np.array(close[:-5]))
    #print logRet_5

    # 计算每5日的指数成交量的对数差，作为特征状态的一个指标
    logVol_5 = np.log(np.array(volume[5:])) - np.log(np.array(volume[:-5]))
    #print logVol_5

    # 由于计算中出现了以5天为单位的计算，所以要调整特征指标的长度。
    logDel = logDel[5:]
    logRet_1 = logRet_1[::4]
    close = close[5:]
    #Date = pd.to_datetime(data.index[5:])
    Date = pd.to_datetime((data['Date'])[5:])
    #print Date
    print close
    Date = np.array(Date)
    print Date

    A = np.column_stack([logDel, logRet_5, logVol_5])
    #print A

    model = GaussianHMM(n_components=n, covariance_type="full", n_iter=2000).fit(A)
    hidden_states = model.predict(A)
    #print hidden_states

    plt.figure(figsize=(25, 18))
    for i in range(model.n_components):
        pos = (hidden_states == i)
        plt.plot_date(Date[pos], close[pos], 'o', label='hidden state %d' % i, lw=2)
        plt.plot_date(close[pos], 'o', label='hidden state %d' % i, lw=2)
        plt.legend(loc="left")







