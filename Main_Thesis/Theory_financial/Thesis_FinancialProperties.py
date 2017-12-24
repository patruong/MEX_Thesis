# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:50:16 2017

@author: extra23
"""

from time import time 
from os import listdir
from os.path import isfile, join
from os import chdir
from os import getcwd

from datetime import datetime, date
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn import preprocessing
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn import cluster
import scipy as sp
#import yahoo_finance #Yahoo finance api - realtime


chdir("C:\\Users\\Patrick\\Documents\\MEX\\MEX\\Local laptop backup\\Other Financial Data\\TDA")
print(getcwd())


# LOW CORR
ALIV = pd.read_csv("ALV.csv", index_col = 0)
dateIndex = pd.DatetimeIndex(ALIV.index)
ALIV.index = dateIndex
ALIV_adj = pd.DataFrame(data = np.log(ALIV["Adj Close"][1:].values/ALIV["Adj Close"][:-1].values), 
             index = ALIV["Adj Close"][1:].index, 
             columns = ["ALIV"])


Chipotle = pd.read_csv("CMG.csv", index_col = 0)
dateIndex = pd.DatetimeIndex(Chipotle.index)
Chipotle.index = dateIndex
Chipotle_adj = pd.DataFrame(data = np.log(Chipotle["Adj Close"][1:].values/Chipotle["Adj Close"][:-1].values), 
             index = Chipotle["Adj Close"][1:].index, 
             columns = ["Chipotle"])

df = pd.concat([ALIV_adj, Chipotle_adj], axis = 1)
df = df['2016']
corr = df.corr()
plt.plot(df["ALIV"], df["Chipotle"], '.')
plt.xlabel("ALIV")
plt.ylabel("CMG")
plt.title("Log Return")

# HIGH CORR
A = pd.read_csv("ATCO-A.ST.csv", index_col = 0, na_values = 'null')
dateIndex = pd.DatetimeIndex(A.index)
A.index = dateIndex
A_adj = pd.DataFrame(data = np.log(A["Adj Close"][1:].values/A["Adj Close"][:-1].values), 
             index = A["Adj Close"][1:].index, 
             columns = ["ATLAS-A"])


B = pd.read_csv("ATCO-B.ST.csv", index_col = 0, na_values = 'null')
dateIndex = pd.DatetimeIndex(B.index)
B.index = dateIndex
B_adj = pd.DataFrame(data = np.log(B["Adj Close"][1:].values/B["Adj Close"][:-1].values), 
             index = B["Adj Close"][1:].index, 
             columns = ["ATLAS-B"])

df = pd.concat([A_adj, B_adj], axis = 1)
df = df['2016']
corr = df.corr()
plt.plot(df["ATLAS-A"], df["ATLAS-B"], '.')
plt.xlabel("ATLAS-A")
plt.ylabel("ATLAS-B")
plt.title("Log Return")


############### PLOT TS
plt.plot(ALIV["Adj Close"])
plt.title("ALIV")
plt.xlim([ALIV.index[0], ALIV.index[-1]])


plt.plot(ALIV_adj)
plt.xlim([ALIV.index[0], ALIV.index[-1]])


############## DISTRIBUTION
from scipy.stats import norm
from scipy.stats import laplace
from scipy.stats import probplot
std = np.std(ALIV_adj)
mean = np.mean(ALIV_adj)

h = sorted(ALIV_adj.values)
h = pd.DataFrame(h)
fit = norm.pdf(h, mean, std)
fit = laplace.pdf(h, 0, 0.014862419499466137)
plt.plot(h,fit, '-')
plt.hist(h, normed = True, bins = 100)
plt.title("ALIV and Laplace distribution")
#plt.hist(h)
plt.show()

import pylab
measurements = np.random.normal(loc = 20, scale = 5, size=100)   
probplot(measurements, dist="norm", plot=pylab)


## Gen random
QQ_aliv = np.random.normal(0, 1, 100)
QQ_aliv = np.random.laplace(2, 5, size = 100)
QQ_aliv = np.random.uniform(-1.6,1.6, size = 100) #(upper-lower, lower, rs) returned by probplot

##
QQ_aliv = ALIV_adj.values.transpose()[0]
probplot(QQ_aliv, dist="norm", plot=pylab)
plt.title("Normal QQ plot")
probplot(QQ_aliv, dist="laplace", plot=pylab)
plt.title("Laplace QQ plot")
probplot(QQ_aliv, dist="uniform", plot=pylab)
plt.title("Laplace QQ plot")

##tuple return (scaling, loc, rss) <-- seem to work

# QQ PLot 
QQ_plt = probplot(QQ_aliv, dist="norm")
QQ_plt = probplot(QQ_aliv, dist="laplace")
plt.plot(QQ_plt[0][0], QQ_plt[0][1], 'bo')
plt.plot(QQ_plt[0][0], QQ_plt[0][0]*QQ_plt[1][0], 'r')

# Sum of squared errors
SSE = ((QQ_plt[0][1] - QQ_plt[0][0]*QQ_plt[1][0])**2).sum()


