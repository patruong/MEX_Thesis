# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 20:29:07 2017

@author: Patrick
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

##############################
### FUNCTIONS DECLARATIONS ###
##############################
from scipy.stats import norm
from scipy.stats import laplace
from scipy.stats import uniform

def emp_d(data, bins = 100, method = 'norm', loc = 0, scaling = np.sqrt(2), title = "Empirical distribution"):
    mean = data.mean()
    std = data.std()
    h = sorted(data)
    h = pd.DataFrame(h)
    if method == 'norm':
        fit = norm.pdf(h, mean, std)
    elif method == 'laplace':
        fit = laplace.pdf(h, loc, scaling)
    elif method == 'uniform':
        fit = uniform.pdf(h)
    plt.plot(h,fit, '-')
    plt.hist(h, normed = True, bins = bins)
    plt.title(title)   

########################
### READIN SECTION #####
########################

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

#READIN QN bin-data
filename = "idq_binary_sample_10_9.bin"

from numpy import fromfile, dtype
# 2 - split for uniform distribution to Laplace
dt = dtype([('x', 'i4'), ('y', 'i4')])
records = fromfile(filename, dt)
df_QN = pd.DataFrame(records)

# 4 - split - for normal distribution to Laplace
dt = dtype([('x', 'i4'), ('y', 'i4'), ('u', 'i4'), ('v','i4')])
records = fromfile(filename, dt)
df_QN = pd.DataFrame(records)

# 8 - split - for uniform to normal distribution to Laplace
dt = dtype([('u1', 'i4'), ('u2', 'i4'), ('u3', 'i4'), ('u4','i4'), ('u5', 'i4')])
records = fromfile(filename, dt)
df_QN = pd.DataFrame(records)

############################################
### STANDARDIZED AND NORMALIZED EURUSD #####
############################################


from sklearn.preprocessing import StandardScaler

dat = df[['logR_ask', 'logR_bid']]
dat = dat.dropna()

#standardization -> this one ie better
scaler = StandardScaler().fit(dat)
rescaledX = scaler.transform(dat)
standardized_df = pd.DataFrame(rescaledX, columns = ['V1'])

#normalization
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(dat)
normalizedX = scaler.transform(dat)
normalized_df = pd.DataFrame(normalizedX, columns = ['logR_ask', 'logR_bid'])



# Inverse transform method / sampling - transform uniform QN data to normal distribution
#https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
#https://se.mathworks.com/matlabcentral/answers/35281-transforming-uniform-variables-to-normal-variables

from sklearn import preprocessing
data = df_QN.values
# normalize to open interval (0,1), not [0,1] because erfinv(1) == inf
min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0,1)) # Uniform to Laplace
min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0.2**(2**4),1-0.2**(2**4))) # 16-digit precision
 
norm_data = min_max_scaler.fit_transform(data)

norm_QN = pd.DataFrame(norm_data, columns = ['x', 'y']) # uniform to Laplace
norm_QN = pd.DataFrame(norm_data, columns = ['x', 'y','u','v']) # normal to Laplace
norm_QN = pd.DataFrame(norm_data, columns = ['x', 'y','u','v']) # normal to Laplace
from scipy import special

# uniform distribution to N(mu, sigma)-distribution
mu = 0
sigma = 2

X = mu + np.sqrt(2)*sigma*special.erfinv(2*norm_QN['x']-1)
Y = mu + np.sqrt(2)*sigma*special.erfinv(2*norm_QN['y']-1)
U = mu + np.sqrt(2)*sigma*special.erfinv(2*norm_QN['u']-1)
V = mu + np.sqrt(2)*sigma*special.erfinv(2*norm_QN['v']-1)

# Laplace from norm-dist L(sqrt(2)) from N(0,1) 
# https://artofproblemsolving.com/community/c2426h1049363_simulating_laplace_random_numbers_from_normal_distribution
# Stock distributions are more "Laplace" than random
# https://sixfigureinvesting.com/2016/03/modeling-stock-market-returns-with-laplace-distribution-instead-of-normal/
scaling = 5
Z = (X*Y - U*V) / scaling # Laplace dist from normal
#Z2 = (X*Y - U*V) / scaling

# Laplace from uniform distribution
# http://businessforecastblog.com/the-laplace-distribution-and-financial-returns/

X = norm_QN['x']
Y = norm_QN['y']

Z = np.log(X/Y) # Laplace dist from uniform


##########################################################
# FOR THESIS 8-split, half to normal half to laplace #####
##########################################################

from sklearn import preprocessing
data = df_QN.values
# normalize to open interval (0,1), not [0,1] because erfinv(1) == inf
min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0,1)) # Uniform to Laplace
min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0.2**(2**4),1-0.2**(2**4))) # 16-digit precision
 
norm_data = min_max_scaler.fit_transform(data)

norm_QN = pd.DataFrame(norm_data, columns = ['u1', 'u2', 'u3', 'u4','u5']) # normal to Laplace
from scipy import special

# uniform distribution to N(mu, sigma)-distribution
mu = 0
sigma = 1

Z = mu + np.sqrt(2)*sigma*special.erfinv(2*norm_QN['u1']-1)
Z1 = mu + np.sqrt(2)*sigma*special.erfinv(2*norm_QN['u2']-1)
Z2 = mu + np.sqrt(2)*sigma*special.erfinv(2*norm_QN['u3']-1)
Z3 = mu + np.sqrt(2)*sigma*special.erfinv(2*norm_QN['u4']-1)
Z4 = mu + np.sqrt(2)*sigma*special.erfinv(2*norm_QN['u5']-1)


# Laplace from norm-dist L(sqrt(2)) from N(0,1) 
# https://artofproblemsolving.com/community/c2426h1049363_simulating_laplace_random_numbers_from_normal_distribution
# Stock distributions are more "Laplace" than random
# https://sixfigureinvesting.com/2016/03/modeling-stock-market-returns-with-laplace-distribution-instead-of-normal/
scaling = 1
V = (Z1*Z2 - Z3*Z4) / scaling # Laplace dist from normal
V_mean = V.mean()
V_std = V.std()
V_standardized = (V - V_mean) / V_std
"""Z ~ N(0,1) and V ~ La(0,1)"""

###
## We want QN noise to be same order of magnitude and same distribution as stock data
##


##############
### QQ PLOT ##
##############


from scipy.stats import probplot
measurements = np.random.normal(loc = 20, scale = 5, size=100)   
probplot(measurements, dist="norm", plot=pylab)

#QQ = np.random.laplace(1, 10, 1000) # Generate laplace data to test
QQ = dat['logR_ask'][0:2000].values
QQ =  standardized_df['logR_ask'][0:2000]
QQ = V[0:2000]
QQ = V_standardized[0:2000]
#QQ = V_s[0:2000].transpose().values[0]
QQ = Z[0:2000]
probplot(QQ, dist="norm", plot=pylab)
plt.title("Normal QQ-plot")
probplot(QQ, dist="laplace", plot=pylab)
plt.title("Laplace QQ-plot")
probplot(QQ, dist="uniform", plot=pylab)
plt.title("Uniform QQ-plot") #(1.6, -1.6)

##tuple return (scaling, loc, rss) <-- seem to work

# QQ PLot 
QQ_plt = probplot(QQ, dist="norm")
QQ_plt = probplot(QQ, dist="laplace")
QQ_plt = probplot(QQ, dist="uniform")
plt.plot(QQ_plt[0][0], QQ_plt[0][1], 'bo')
plt.plot(QQ_plt[0][0], QQ_plt[0][0]*QQ_plt[1][0], 'r')

# Sum of squared errors
SSE = ((QQ_plt[0][1] - QQ_plt[0][0]*QQ_plt[1][0])**2).sum()
SSE

######################
## SCALING FINDER ####
######################

data = QQ
mean = data.mean()
std = data.std()
h = sorted(data)
h = pd.DataFrame(h)
fit = norm.pdf(h, mean, std)

#fit = laplace.pdf(h, loc, scaling)
plt.plot(h,fit, '-')
temp = plt.hist(h, normed = True, bins = 100)
plt.title("Persistent integral distribution")  

fit.transpose()[0]


import matplotlib.pyplot as plt

lines = plt.plot([1,2,3],[4,5,6],[7,8],[9,10])
lines[0].get_data()
lines[1].get_data()
