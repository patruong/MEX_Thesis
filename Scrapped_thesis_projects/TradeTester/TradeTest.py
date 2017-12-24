# -*- coding: utf-8 -*-
"""
Created on Sat May  6 20:43:31 2017

@author: Patrick
"""

from os import listdir
from os.path import isfile, join
from os import chdir
from os import getcwd
chdir("C:\\Users\\Patrick\\Documents\\data\\Other Financial Data\\OMX30_01012000-06052017")
print(getcwd())

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn import preprocessing
#import yahoo_finance #Yahoo finance api - realtime

from pandas.io.data import DataReader
from pandas_datareader.data import DataReader # Read data directly from Yahoo Finance
from datetime import datetime
"""
Good post about pandas.io.data
http://stackoverflow.com/questions/12433076/download-history-stock-prices-automatically-from-yahoo-finance-in-python

"""

omx = DataReader('^OMX',  'yahoo', datetime(2000, 1, 1), datetime(2017, 5, 5))
omx_adj = omx['Adj Close']

ma20 = pd.rolling_mean(omx_adj, window = 20)
ma20.name = 'ma20'
ma50 = pd.rolling_mean(omx_adj, window = 50)
ma50.name = 'ma50'
ma200 = pd.rolling_mean(omx_adj, window = 200)
ma200.name = 'ma200'

ax1 = omx_adj.plot()#(legend = 'Adj close')
ax1 = ma20.plot()#(legend = 'ma20')
ma50.plot()#(legend = 'ma50')
ma200.plot()#(legend = 'ma200')
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines[:], labels[:], loc='best')

"BACK TESTING"
t1 = (ma50[:-1] > ma200[:-1]).values
t2 = (ma50[1:] > ma200[1:]).values
plt.plot(t1 == t2)

df_ma50ma200 = pd.DataFrame(data = t1, index = omx.index[:-1], columns = ['ma50>ma200'] )
df_SQ = pd.DataFrame(data = (t1==t2), index = omx.index[:-1], columns = ['StatusQuo'])
df_test = pd.concat([omx_adj[:-1], df_ma50ma200, df_SQ], axis = 1)[199:] #199 because this is where ma200 begins

df_test['2011-07'] # True-False ==> Sälj
df_test['2012-2'] # False-False ==> köp

#df_test[df_test['ma50>ma200'] == False and df_test['StatusQuo'] == False]

"Find all change points"
df_change = df_test[df_test['StatusQuo'] == False]

"Buy for all your cash . ma50 > ma200 strat"
Own = False
Capital = 3000
n_stock = 0
value = []
dates = []
for i in range(len(df_change)):
    #print(df_change.iloc[i])
    if ((df_change.iloc[i]["ma50>ma200"] == False) and 
        (df_change.iloc[i]['StatusQuo'] == False)):
        Own = True
        while (Capital > n_stock * df_change.iloc[i]["Adj Close"]):
            n_stock += 1
            if (Capital < n_stock * df_change.iloc[i]["Adj Close"]):
                n_stock -= 1
                break
        #Capital = Capital - (Capital%df_change.iloc[i]["Adj Close"])
        Capital = (Capital - n_stock*df_change.iloc[i]["Adj Close"])
        
        print("BROUGHT stock at " + str(df_change.iloc[i].name.date()))
        print("Stocks: " + str(n_stock))
        print("Stock Value: " + str(n_stock*df_change.iloc[i]["Adj Close"]))
        print("Capital: " + str(Capital))
        print("Total Value: " + str(Capital + 
                                    n_stock*df_change.iloc[i]["Adj Close"]))
        print()
        
        value.append(Capital + n_stock*df_change.iloc[i]["Adj Close"])
        dates.append(df_change.iloc[i].name.date())
    
    if ((Own == True) and (df_change.iloc[i]["ma50>ma200"] == True) and
        (df_change.iloc[i]["StatusQuo"] == False)):
        Own = False
        Capital = (Capital + df_change.iloc[i]["Adj Close"]*n_stock)
        n_stock = 0
        
        print("SOLD stock at " + str(df_change.iloc[i].name.date()))
        print("Capital: " + str(Capital))
        print()
        
        value.append(Capital)
        dates.append(df_change.iloc[i].name.date())
        
print("Stocks: " + str(n_stock))
print("Stock Value: " + str(n_stock * df_change.iloc[i]["Adj Close"]))
print("Capital: " + str(Capital))
print("Total Value: " + str(Capital + n_stock * df_change.iloc[i]["Adj Close"]))
plt.step(dates, value)
        
                                          
""" MA20 MA50 """

"BACK TESTING"
t1 = (ma20[:-1] > ma50[:-1]).values
t2 = (ma20[1:] > ma50[1:]).values
plt.plot(t1 == t2)

df_ma20ma50 = pd.DataFrame(data = t1, index = omx.index[:-1], columns = ['ma20>ma50'] )
df_SQ = pd.DataFrame(data = (t1==t2), index = omx.index[:-1], columns = ['StatusQuo'])
df_test = pd.concat([omx_adj[:-1], df_ma20ma50, df_SQ], axis = 1)[49:] #49 because this is where ma50 begins


#df_test[df_test['ma20>ma50'] == False and df_test['StatusQuo'] == False]

"Find all change points"
df_change = df_test[df_test['StatusQuo'] == False]

"Buy for all your cash . ma20 > ma50 strat"
Own = False
Capital = 3000
n_stock = 0

# STORED INFO
value_list = []
dates_list = []
stocks_list = []
capital_list = []
buy_list = []

for i in range(len(df_change)):
    #print(df_change.iloc[i])
    if ((df_change.iloc[i]["ma20>ma50"] == False) and 
        (df_change.iloc[i]['StatusQuo'] == False)):
        Own = True
        while (Capital > n_stock * df_change.iloc[i]["Adj Close"]):
            n_stock += 1
            if (Capital < n_stock * df_change.iloc[i]["Adj Close"]):
                n_stock -= 1
                break
        #Capital = Capital - (Capital%df_change.iloc[i]["Adj Close"])
        Capital = (Capital - n_stock*df_change.iloc[i]["Adj Close"])
        
        print("BROUGHT stock at " + str(df_change.iloc[i].name.date()))
        print("Stocks: " + str(n_stock))
        print("Stock Value: " + str(n_stock*df_change.iloc[i]["Adj Close"]))
        print("Capital: " + str(Capital))
        print("Total Value: " + str(Capital + 
                                    n_stock*df_change.iloc[i]["Adj Close"]))
        print()
        
        # Store Info
        value_list.append(Capital + n_stock*df_change.iloc[i]["Adj Close"])
        dates_list.append(df_change.iloc[i].name.date())
        stocks_list.append(n_stock)
        capital_list.append(Capital)
        buy_list.append(True)
    
    if ((Own == True) and (df_change.iloc[i]["ma20>ma50"] == True) and
        (df_change.iloc[i]["StatusQuo"] == False)):
        Own = False
        Capital = (Capital + df_change.iloc[i]["Adj Close"]*n_stock)
        n_stock = 0
        
        print("SOLD stock at " + str(df_change.iloc[i].name.date()))
        print("Capital: " + str(Capital))
        print()
        
        # Store info
        value_list.append(Capital)
        dates_list.append(df_change.iloc[i].name.date())
        stocks_list.append(n_stock)
        capital_list.append(Capital)
        buy_list.append(False)
        
print("Stocks: " + str(n_stock))
print("Stock Value: " + str(n_stock * df_change.iloc[i]["Adj Close"]))
print("Capital: " + str(Capital))
print("Total Value: " + str(Capital + n_stock * df_change.iloc[i]["Adj Close"]))
plt.step(dates_list, value_list)


"CREATE INFO DF"
zipped = zip(tuple(capital_list), tuple(stocks_list), tuple(buy_list))
#zipped = zip(capital_list, stocks_list)
#info_data = [tuple(capital_list),
#             tuple(stocks_list),
#                  tuple(buy_list)]
info_data = list(zipped)
labels  = ['Capital', 'n_stocks', "buy"]
df_info = pd.DataFrame(data = info_data, index = dates_list, columns = labels)

df_portfolio = pd.concat([omx_adj, df_info], axis = 1)
df_portfolio.fillna(method = 'ffill', inplace = True)
df_portfolio['Stock value'] = df_portfolio['n_stocks']*df_portfolio["Adj Close"]
df_portfolio['Total value'] = df_portfolio['Stock value'] + df_portfolio['Capital']

df_portfolio["Total value"].plot()
plt.step(dates_list, value_list)

omx_adj['2017-05-05']/omx_adj['2009-02-26']
df_portfolio['Total value']['2017-05-05']/df_portfolio['Total value']['2009-02-26']

