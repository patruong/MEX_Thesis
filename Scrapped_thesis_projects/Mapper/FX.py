# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:02:57 2017

@author: extra23
"""


from time import time 

from os import listdir
from os.path import isfile, join
from os import chdir
from os import getcwd
#chdir("C:\\Users\\Patrick\\Documents\\data\\Other Financial Data\\TDA")
chdir("C:\\Users\\extra23\\Desktop\\Filer\\MEX\\Local laptop backup\\Other Financial Data\\TDA")
print(getcwd())

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn import preprocessing

from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn import cluster
import scipy as sp


# READ IN DATA
df = pd.read_csv("FX.csv")
len(df.columns)

# DF SORTING
df_list = [] # List of each forex
for i in range(len(df.columns)):
    if (i % 2 == 0):
        
        df_temp = pd.DataFrame(df.ix[:,i+1].values)
        #print([(df.ix[:,i:i+2].columns[1][:-12])[-3:]])
        str_temp = (df.ix[:,i:i+2].columns[1][:-12])[:3]
        str_temp2 = (df.ix[:,i:i+2].columns[1][:-12])[-3:]
        df_temp.columns = [str_temp + "/" + str_temp2]
        df_temp.index = df.ix[:,i].values
        # Remake index rows to dateIndex
        dateIndex = pd.DatetimeIndex(df_temp.index)
        df_temp.index = dateIndex
        df_temp = df_temp.dropna()
        
        df_list.append(df_temp)

df_cum = pd.DataFrame()
for i in df_list:
    df_cum = pd.concat([df_cum, i], axis = 1)
   
    
#FILLNA
df_cum = df_cum.fillna(method = "ffill")

# CALCULATE LOG-RETURNS
df_log = pd.DataFrame(data = np.log(df_cum[1:].values / df_cum[:-1].values), index = df_cum[:-1].index, columns = df_cum.columns)
#df_adjClose_diff.fillna(0) # make NaN zero
df_log = df_log.fillna(method = "ffill")

df_cum.to_csv("FX_raw.csv")
df_log.to_csv("FX_log.csv")
