# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:52:52 2017

@author: PTruong

Script to build one dataframe from many different stock time series

"""


from os import listdir
from os.path import isfile, join
from os import chdir
from os import getcwd
chdir("C:\\Users\\extra23\\Desktop\\Scripts\\stock")
print(getcwd())

import numpy as np
import pandas as pd
import scipy as sp

    
" READ IN OF FILES IN FOLDER "
df = pd.DataFrame()

for f in listdir():
    # Read in csv 
    #df = pd.read_csv(f, index_col = 0, na_values = 'null')
    df_stock = pd.read_csv(f, index_col = 0)
    
    #Calculate Spread
    df_spread = pd.DataFrame(df_stock['High']-df_stock['Low'], columns = ['Spread'])
    df_stock = pd.concat([df_stock, df_spread], axis = 1)
    
    # Remake index rows to dateIndex
    dateIndex = pd.DatetimeIndex(df_stock.index)
    df_stock.index = dateIndex
    
    # Create temp variable for current read file
    #df_temp = pd.DataFrame(data = df["Adj Close"])
    #df_temp.columns = [file_list[f][:-4]]
    
    # Create multiIndex
    tuples = list(zip(*[[f[:-4] for z in range(len(df_stock.columns))], df_stock.columns]))
    index = pd.MultiIndex.from_tuples(tuples)
    df_stock.columns = index
    
    # Fill our stock matrix
    if df.empty == True:
        df = df.append(df_stock)
    else:
        df = pd.concat([df, df_stock], axis = 1)
        
"""
df_stock1 = pd.read_csv("ZTS.csv", index_col = 0)
df_spread = pd.DataFrame(df_stock1['High']-df_stock1['Low'], columns = ['Spread'])
df_stock1 = pd.concat([df_stock1, df_spread], axis = 1)
dateIndex = pd.DatetimeIndex(df_stock1.index)
df_stock1.inedx = dateIndex
df_stock1.columns = df_stock1.columns
#[['Z' for z in range(len(df_stock1.columns))], df_stock1.columns]
tuples = list(zip(*[['Z' for z in range(len(df_stock1.columns))], df_stock1.columns]))
index = pd.MultiIndex.from_tuples(tuples)
df_stock1.columns = index

df_stock2 = pd.read_csv("YUM.csv", index_col = 0)
df_spread = pd.DataFrame(df_stock2['High']-df_stock2['Low'], columns = ['Spread'])
df_stock2 = pd.concat([df_stock2, df_spread], axis = 1)
dateIndex = pd.DatetimeIndex(df_stock1.index)
df_stock2.inedx = dateIndex
df_stock2.columns = df_stock2.columns
#[['Z' for z in range(len(df_stock1.columns))], df_stock1.columns]
tuples = list(zip(*[['Y' for z in range(len(df_stock2.columns))], df_stock2.columns]))
index = pd.MultiIndex.from_tuples(tuples)
df_stock2.columns = index

df_s = pd.concat([df_stock1, df_stock2], axis = 1)
"""

# Some indexing
df.iloc[:, df.columns.get_level_values(1) == 'Close']
df.iloc[:, df.columns.get_level_values(1) == 'Close'].ix['2014':'2015']


