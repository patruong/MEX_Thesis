# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:26:20 2017

@author: Patrick


FILE FOR MERGING AND NORMLAIZING THINGS


"""

from os import listdir
from os.path import isfile, join
from os import chdir
from os import getcwd
import numpy as np
import pandas as pd
from pyentrp import entropy as ent
import matplotlib.pyplot as plt
path = "C:\\Users\\Patrick\\Documents\\MEX\\MEX\\Local laptop backup\\Other Financial Data\\TDA\\Paris_Intraday"
chdir(path)

#############
############
#############
# MAIN READIN
df = pd.read_csv('EURUSD_20170814-20170818.csv', usecols = [1,2,3], index_col = 0)
df_u = pd.read_csv('EURUSD_20170814-20170818_UnixTS.csv', usecols = [1,2,3], index_col = 0)
df_QN = pd.read_csv('QN_a.csv')

# adding additional information
df['unix'] = df_u.index
df['logR_bid'] = np.log(df.bid) - np.log(df.bid.shift(1))
df['logR_ask'] = np.log(df.ask) - np.log(df.ask.shift(1))
df['spread'] = df.ask - df.bid

#reorder columns
df = df[['unix', 'bid', 'ask', 'spread', 'logR_ask', 'logR_bid']]
############
############
############
from sklearn.preprocessing import StandardScaler

dat = df[['logR_ask', 'logR_bid']]
dat = dat.dropna()

#standardization -> this one ie better
scaler = StandardScaler().fit(dat)
rescaledX = scaler.transform(dat)
rescaledX_df = pd.DataFrame(rescaledX, columns = ['logR_ask', 'logR_bid'])

#normalization
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(dat)
normalizedX = scaler.transform(dat)
normalizedX_df = pd.DataFrame(normalizedX, columns = ['logR_ask', 'logR_bid'])

filename = "idq_binary_sample_10_9.bin"

dt = np.dtype([('a', 'i4'), ('b', 'i4'), ('c', 'i4'), ('d', 'f4'), ('e', 'i4'),
               ('f', 'i4', (256, ))])
data = np.fromfile(filename, dtype=dt)

df = pd.DataFrame(data.tolist(), columns = data.dtype.names)
sample_df = df['a'][0:2000]

df_roll = rolling_window(df['a'], 2000, 2000)

"other example"
from numpy import fromfile, dtype
from pandas import DataFrame
dt = dtype([('name', 'S'), ('data1', 'int32'), ('data2', 'float64')])
records = fromfile(filename, dt)
df_2 = DataFrame(records)


from numpy import fromfile, dtype
from pandas import DataFrame
dt = dtype([('a', 'i4'), ('b', 'i4')])
records = fromfile(filename, dt)
df = DataFrame(records)

with open(filename, "rb") as binary_file:
    # Read the whole file at once
    data = binary_file.read()

#Read in as bytes and convert to binary values
data = np.fromfile(filename, dtype='B')
boolean_array = (data > 127)
bin_array = boolean_array*1 

#Read in as integers
data = np.fromfile(filename, dtype='int32')



# Laplace from norm-dist
# https://artofproblemsolving.com/community/c2426h1049363_simulating_laplace_random_numbers_from_normal_distribution
from numpy import fromfile, dtype
from pandas import DataFrame
dt = dtype([('x', 'i4'), ('y', 'i4'), ('c', 'i4'), ('d', 'i4')])
#dt = dtype([('a', 'i4'), ('b', 'i4')])
records = fromfile(filename, dt)
df = DataFrame(records)

# Inverse transform method / sampling 
#https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
#https://se.mathworks.com/matlabcentral/answers/35281-transforming-uniform-variables-to-normal-variables

from sklearn import preprocessing
data = df.values
min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0.2**(2**4),1-0.2**(2**4)))
norm_data = min_max_scaler.fit_transform(data)
norm_df = pd.DataFrame(norm_data, columns = ['a', 'b'])
norm_df = pd.DataFrame(norm_data, columns = ['a', 'b','c','d'])
from scipy import special
mu = 0
sigma = 1

Y = mu + np.sqrt(2)*sigma*special.erfinv(2*norm_df['a']-1)


###
### empriical distributions
###
from scipy.stats import norm

# uniform distribution
mean = df['a'][0:100000].mean()
std = df['a'][0:10000].std()
h = sorted(df['a'][0:100000])
h = pd.DataFrame(h)
fit = norm.pdf(h, mean, std)
plt.plot(h,fit, '-')
plt.hist(h, normed = True, bins = 100)
plt.title("Persistent integral distribution")  

# Normal dist generated from uniform
# Y[8210000:8220000].plot() - somewhere between is inf value with bound at 0.1**(2**4)
mean = Y[0:100000].mean()
std = Y[0:100000].std()
h = sorted(Y[0:100000])
h = pd.DataFrame(h)
fit = norm.pdf(h, mean, std)
plt.plot(h,fit, '-')
plt.hist(h, normed = True, bins = 100)
plt.title("Persistent integral distribution")   

####
mean = rescaledX_df['logR_ask'][1000:1200].mean()
std = rescaledX_df['logR_ask'][1000:1200].std()
h = sorted(rescaledX_df['logR_ask'][1000:1200])
h = pd.DataFrame(h)
fit = norm.pdf(h, mean, std)
plt.plot(h,fit, '-')
plt.hist(h, normed = True, bins = 100)
plt.title("Persistent integral distribution")   

def emp_d(data):
    mean = data.mean()
    std = data.std()
    h = sorted(data)
    h = pd.DataFrame(h)
    fit = norm.pdf(h, mean, std)
    plt.plot(h,fit, '-')
    plt.hist(h, normed = True, bins = 100)
    plt.title("Persistent integral distribution")   



def emp_dist(data, start = False, end = False):
    if (start == False and end == False):
        mean = data.mean()
        std = data.std()
    elif(start == False):
        mean = data[start:].mean()
        std = data[start:].std()
    elif(end == False):
        mean = data[:end].mean()
        std = data[:end].std()
    else:
        mean = data[start:end].mean()
        std = data[start:end].mean()
    h = sorted(data)
    h = pd.DataFrame(h)
    fit = norm.pdf(h, mean, std)
    plt.plot(h,fit, '-')
    plt.hist(h, normed = True, bins = 100)
    plt.title("Persistent integral distribution")   
