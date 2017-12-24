# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:17:02 2017

@author: Patrick
"""
from os import listdir
from os.path import isfile, join
from os import chdir
from os import getcwd
#chdir("C:\\Users\\Patrick\\Documents\\data\\Other Financial Data\\OMX30_01012000-06052017")
chdir("C:\\Users\\extra23\\Desktop\\Filer\\MEX\\Local laptop backup\\Other Financial Data\\OMX30_01012000-06052017")
print(getcwd())

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn import preprocessing

" CREATE LIST OF FILES IN FOLDER"
file_list = []
for f in range(len(listdir())):
    #if (listdir()[f][0:5] == "green") and (listdir()[f][-4:] == '.csv'):
    #    file_list.append(listdir()[f])
    if (listdir()[f][-3:] != '.py'):
        file_list.append(listdir()[f])

" READ IN OF FILES IN FOLDER "
df_adjClose = pd.DataFrame()
for f in range(len(file_list)):
    # Read in csv 
    df = pd.read_csv(file_list[f], index_col = 0, na_values = 'null')
    
    # Remake index rows to dateIndex
    dateIndex = pd.DatetimeIndex(df.index)
    df.index = dateIndex
    
    # Create temp variable for current read file
    df_temp = pd.DataFrame(data = df["Adj Close"])
    df_temp.columns = [file_list[f][:-4]]
    
    # Fill our stock matrix
    if df_adjClose.empty == True:
        df_adjClose = df_adjClose.append(df_temp)
    else:
        df_adjClose = pd.concat([df_adjClose, df_temp], axis = 1)

# two types of normalization - USEFUL?
norm_meanStd = (df_adjClose - df_adjClose.mean())/df_adjClose.std()
norm_minMax = (df_adjClose-df_adjClose.min())/(df_adjClose.max()-df_adjClose.min())

" CALCULATE PERCENT DIFF"
#df_adjClose[1:] # t+1
#df_adjClose[:-1] # t
#df_adjClose_diff = df_adjClose[1:] / df_adjClose[:-1]

df_adjClose = df_adjClose.fillna(method = "ffill")
df_adjClose_diff = pd.DataFrame(data = np.log(df_adjClose[1:].values / df_adjClose[:-1].values), index = df_adjClose[:-1].index, columns = df_adjClose.columns)
df_adjClose_diff.fillna(0) # make NaN zero

" Plot normalized return curves "
np.exp(df_adjClose_diff.cumsum()).plot()


df_adjClose_diff['INVE-B.ST']["2015-04-09":"2017-05-04"].fillna(0).mean()
df_adjClose['INVE-B.ST']["2015-04-09":"2017-05-04"].plot()
