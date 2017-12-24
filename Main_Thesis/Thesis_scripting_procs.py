# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 14:04:34 2017

@author: Patrick
"""

"""
CONTAINS VARIOUS SCRIPTING PROCEDURE FOR THE THESIS INCLUDING:
    - COMPLEXITY CALCULATIONS
    - POWER SPECTRAL DENSITY
    - FALSE NEAREST NEIGHBORS CALCULATIONS
    - CLUSTERING ON LANDSCAPES
    - etc.


NOTE TO SELF: do NOT convert to real timestamps until last step, if before
last step, the dates will become strings when going from df to series etc.

Also all computational speed is increased with Unix timestamp

"""
from os import listdir
from os.path import isfile, join
from os import chdir
from os import getcwd
path = "C:\\Users\\Patrick\\Documents\\MEX\\MEX\\Local laptop backup\\Other Financial Data\\TDA\\Paris_intraday" 
#path = "C:\\Users\\Patrick\\Documents\\MEX\\MEX\\Local laptop backup\\Other Financial Data\\TDA\\Paris_intraday\\C:\Users\Patrick\Documents\MEX\MEX\Local laptop backup\Other Financial Data\TDA\Paris_Intraday\\Final Results CSV" 
path = "C:\\Users\\Patrick\\Documents\\MEX\\MEX\\Local laptop backup\\Other Financial Data\\TDA\\Paris_Intraday\\Final Results CSV"
chdir(path)
print(getcwd())
 
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import norm #for empricial dist plotting
import nolds # Non-Linear Measures for Dynamical Systems
from pyentrp import entropy as ent #entropy calculations
from lempel_ziv_complexity import lempel_ziv_complexity # for lempel_ziv_complexity
import pypsr #FALSE NEAREST NEIGHBOR AND PHASE STATE RECONST
from statsmodels.tsa.stattools import acf # for ACF function
import scipy.cluster


#Function to time processes
def timeIt(proc):
    start_time_main = time.time()
    res = proc
    print(res)
    print("--- %s seconds ---" % (time.time() - start_time_main))
    
#Rolling window that gives all windows items
# https://stackoverflow.com/questions/39374020/pandas-can-you-access-rolling-window-items
def rolling_window(a, step, jump = 1):
    shape   = a.shape[:-1] + (a.shape[-1] - step + 1, step)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[0::jump] #modified with jump / slicing start:stop:step


def empirical_dist(x, title_txt = "title", n_bins = 100, x_lim = None, y_lim = None):
    """
    x - takes the return time series
    x_lim = (x_min, x_max) tuple
    """
    
    std = np.std(x)
    mean = np.mean(x)
    
    h = sorted(x.values)
    h = pd.DataFrame(h)
    h = h.dropna()
    fit = norm.pdf(h, mean, std)
    plt.plot(h,fit, '-')
    plt.hist(h, normed = True, bins = n_bins)
    plt.title(title_txt)
    if x_lim != None:
        plt.xlim(x_lim)
    if y_lim != None:
        plt.ylim(y_lim)
    plt.show()

import zlib, sys, time, base64
def gzip_compress_ratio(x, ratio = 9):
# Gzip compression ratio
    np.set_printoptions(threshold = 'nan')
    raw = np.array_str(x)
    raw_size = sys.getsizeof(raw)
    compressed = base64.b64encode(zlib.compress(raw, ratio))
    #print(compressed)
    comp_size = sys.getsizeof(compressed)
    #decomp = zlib.decompress(base64.b64decode(compressed))
    #sys.getsizeof(decomp)
    comp_ratio = float(comp_size) / float(raw_size)
    np.set_printoptions(threshold = 10)
    return comp_ratio

    
# MAIN READIN
df = pd.read_csv('EURUSD_20170814-20170818.csv', usecols = [1,2,3], index_col = 0)
df_u = pd.read_csv('EURUSD_20170814-20170818_UnixTS.csv', usecols = [1,2,3], index_col = 0)
df_QN = pd.read_csv('QN_a.csv')
df_std = pd.read_csv('standardized_EURUSD.csv')
df_QN_laplace_std = pd.read_csv("Standardized_Laplace_QuantumNoise_short.csv", header = None)
df_QN_laplace_std = pd.read_csv("Standardized_Laplace_QuantumNoise_short_discrete77.csv", header = None)
# adding additional information
df['unix'] = df_u.index
df['logR_bid'] = np.log(df.bid) - np.log(df.bid.shift(1))
df['logR_ask'] = np.log(df.ask) - np.log(df.ask.shift(1))
df['spread'] = df.ask - df.bid

#reorder columns
df = df[['unix', 'bid', 'ask', 'spread', 'logR_ask', 'logR_bid']]

# DAILY READIN
df_daily = pd.read_csv("FX.csv", usecols = [12,13], index_col = 0)
df_daily['logR'] = np.log(df_daily) - np.log(df_daily.shift(1))
  
"Empirical distribution plot"
#empirical_dist(df['logR_bid'], n_bins = 5000)
#empirical_dist(df['logR_bid'], n_bins = 5000, x_lim = (-0.000025, 0.000025)) # 
#empirical_dist(df_daily['logR'])

################## SW IMPLEMENTATION

"""
NOTE on selecting embedding dimension article:
    Selection of Embedding Dimension and Delay Time in Phase Space Reconstruction

We first implement when going through all dimensions, then we use methods from this and

article : Practicle Implementation of Nonlinear time series methods: the TISEAN package

- False nearest neighbors
- autocorrelation

"""
# 100 dp sliding windows with 10 step jump between each window to save space
window_size = 100
window_size = 2000
emb_dim = 4
rolling = rolling_window(df.logR_ask, window_size, 10)
rolling = rolling_window(df_std.logR_ask, window_size, window_size)
rolling = rolling_window(df_QN_laplace_std.values.transpose()[0], window_size, window_size)
rolling_ns = rolling_window(df.ask, window_size, 10)
rolling_ts = rolling_window(df.index, window_size, 10)
df_ = pd.DataFrame(rolling)

sw_1 = rolling[1]
sw_1_ns = rolling[1]
nolds.lyap_r(sw_1, emb_dim = emb_dim)
nolds.lyap_e(sw_1, emb_dim = emb_dim)
nolds.sampen(sw_1, emb_dim= emb_dim)
nolds.hurst_rs(sw_1)
nolds.corr_dim(sw_1, emb_dim=emb_dim)
nolds.dfa(sw_1)
ent.shannon_entropy(sw_1) # is this even valid? we do not have any p_i states i ALSO IGNORES TEMPORAL ORDER - Practical consideration of permutation entropy
ent.sample_entropy(sw_1, sample_length = 10) #what is sample length?
#ent.multiscale_entropy(sw_1, sample_length = 10, tolerance = 0.1*np.std(sw_1)) # what is tolerance?

                      "Practical considerations of permutation entropy: A Tutorial review - how to choose parameters in permutation entropy"
ent.permutation_entropy(sw_1, m=8, delay = emd_dim )  #Reference paper above 
#ent.composite_multiscale_entropy()
lempel_ziv_complexity(sw_1)
gzip_compress_ratio(sw_1_ns, 9)


#https://www.researchgate.net/post/How_can_we_find_out_which_value_of_embedding_dimensions_is_more_accurate
#when choosing emb_dim for Takens, each dimension should have at least 10 dp ==> 10^1 == 1D, 10^2 == 2D, ..., 10^6 == 6D 

#FALSE NEAREST NEIGHBOR FOR DETERMINING MINIMAL EMBEDDING DIMENSION

#MEASURES OF COMPLEXITY
# https://hackaday.io/project/707-complexity-of-a-time-series

# general entropy with discrete pdf - [H = sum_i - p_i * log( p_i)] , we cannot use because we have not well defined states
# Approximate entropy 


# Recurrency plot
from pyunicorn.timeseries import RecurrenceNetwork
x = np.sin(np.linspace(0, 10*np.pi, 1000))
net = RecurrenceNetwork(x, recurrence_rate = 0.05)

# NOTE WE CAN ALWAYS MAKE ALL THE ROLLING ITEMS IN PY AND ANALYSE IN R!



########

# Parameters
window_size = 100
emb_dim = 4
rolling = rolling_window(df.logR_ask.dropna(), window_size, 10) #dropped NaN from logR
#rolling_ns = rolling_window(df.ask, window_size, 10)
#rolling_ts = rolling_window(df.index, window_size, 10)

df_ = pd.DataFrame(rolling)

#sw_1 = rolling[1]
#sw_1_ns = rolling[1]

nolds.lyap_r(df_1, emb_dim = emb_dim)
lyapunov_r = np.vectorize(nolds.lyap_r)
res_array = lyapunov_r(rolling[0:10], emb_dim) #maybe because vectorized goes though each element not row

res_array = np.zeros(len(rolling))
iter_i = 0
start_time_main = time.time()
for i in rolling[0:20]: #when rolling[0] same error as vectorized maybe it goies trhough each element not row
    res_array[iter_i] = lempel_ziv_complexity(i)
    iter_i += 1
print("--- %s seconds ---" % (time.time() - start_time_main))


sw_1 = rolling[1]
nolds.lyap_r(sw_1, emb_dim = emb_dim)


#below are just different functions
nolds.lyap_r(i,emb_dim = emb_dim) # 0.8s per calculation
nolds.lyap_e(sw_1, emb_dim = emb_dim) # slower than lyap_r
nolds.sampen(sw_1, emb_dim= emb_dim) #quite fast?
nolds.hurst_rs(sw_1) # > 1s per calculation
nolds.corr_dim(sw_1, emb_dim=emb_dim) # 6 s
nolds.dfa(sw_1) #6 s

#Shannon is fast
ent.shannon_entropy(sw_1) # is this even valid? we do not have any p_i states i ALSO IGNORES TEMPORAL ORDER - Practical consideration of permutation entropy


ent.sample_entropy(sw_1, sample_length = 10) #what is sample length? #153s

#ent.multiscale_entropy(sw_1, sample_length = 10, tolerance = 0.1*np.std(sw_1)) # what is tolerance?

                      "Practical considerations of permutation entropy: A Tutorial review - how to choose parameters in permutation entropy"
start_time_main = time.time()
#ent.permutation_entropy(sw_1, m=8, delay = emb_dim )  #Reference paper above  #2489 seconds
#ent.permutation_entropy(sw_1, m=3, delay = 1 )
nolds.sampen(sw_1, emb_dim= emb_dim)
print("--- %s seconds ---" % (time.time() - start_time_main))

#an increase in m increases computation time alot

start_time_main = time.time()
ent.permutation_entropy(sw_1, m=4, delay = 1 ) #as per "practical consideration" article
print("--- %s seconds ---" % (time.time() - start_time_main)) 

#ent.composite_multiscale_entropy()
start_time_main = time.time()
lempel_ziv_complexity(sw_1) #1s

start_time_main = time.time()
gzip_compress_ratio(sw_1_ns, 9) #0.23s
print("--- %s seconds ---" % (time.time() - start_time_main))  #1s

### Permutatoin entropy rolling window
# Parameters
window_size = 2000
emb_dim = 4
rolling = rolling_window(df.logR_ask.dropna(), window_size, 400) #dropped NaN from logR
rolling = rolling_window(df.logR_ask.dropna(), window_size, 2000)
#rolling_ns = rolling_window(df.ask, window_size, 10)
rolling_ts = rolling_window(df.index, window_size, 400)

df_ = pd.DataFrame(rolling)

sw_1 = rolling[1]
#sw_1_ns = rolling[1]

#nolds.lyap_r(df_1, emb_dim = emb_dim)
#lyapunov_r = np.vectorize(nolds.lyap_r) # vectorized is essentially a for-loop
#res_array = lyapunov_r(rolling[0:10], emb_dim) #maybe because vectorized goes though each element not row

res_array = np.zeros(len(rolling))
iter_i = 0
start_time_main = time.time()
for i in rolling: #when rolling[0] same error as vectorized maybe it goies trhough each element not row
    res_array[iter_i] = lempel_ziv_complexity(i)
    iter_i += 1
print("--- %s seconds ---" % (time.time() - start_time_main))

shannon = np.genfromtxt("shannon_WindowS2000_Gap400.csv", delimiter=',')
gzip_ent = np.genfromtxt("gzip_WindowS2000_Gap400.csv", delimiter=',')
plt.plot(shannon)
plt.title("Shannon Entropy")
plt.plot(gzip_ent)
plt.title("Gzip Compression Ratio")

"""
NOTES: 
    Window_size = 2000
    gap = 400
    Shannon -> 823s
    gzip -> 3930.575 s 
"""

start_time_main = time.time()
ent.permutation_entropy(sw_1, m=3, delay = 1 )
print("--- %s seconds ---" % (time.time() - start_time_main))

start_time_main = time.time()
40*2
print("--- %s seconds ---" % (time.time() - start_time_main))



###########################################

# periodicity of the whole data:
df_logR_ask = df['logR_ask'].dropna()

x = df_std['logR_ask'][0:2000]
start_time_main = time.time()
periodogram = sp.signal.periodogram(df_logR_ask) # ==> non periodic
welch = sp.signal.welch(df_logR_ask) # ==> non periodic
periodogram = sp.signal.periodogram(df_std['logR_ask']) # ==> non periodic
welch = sp.signal.welch(df_std['logR_ask'][0:200]) # ==> non periodic

plt.semilogy(welch[0], welch[1])
#plt.plot(welch[0], welch[1])
#plt.semilogy(welch[0], np.sqrt(welch[1]))
#plt.semilogy(f, np.sqrt(Pxx_spec))
plt.xlabel('frequency [Hz]')
plt.ylabel('Power Spectral Density')
plt.title('Power Spectral Density of EURUSD')

print("--- %s seconds ---" % (time.time() - start_time_main))    



fs = 10e3
N = 1e5
amp = 2*np.sqrt(2)
freq = 1234.0
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
x = amp*np.sin(2*np.pi*freq*time)
x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)

f, Pxx_den = sp.signal.welch(x, fs, nperseg=1024)
plt.semilogy(f, Pxx_den)
plt.ylim([0.5e-3, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')

# For autocorrelation to determine time-delay
def plot_acf(x, nlags = 40):
    """
    plt.plot(acf(sw_1))
    plt.plot([0,len(acf(sw_1))],[0.1, 0.1], '--', color = 'orange')
    plt.plot([0,len(acf(sw_1))],[-0.1, -0.1], '--', color = 'orange' )
    """
    
    plt.plot(acf(x, nlags = 40))
    plt.plot([0, nlags +1], [0.1, 0.1], '--', color = 'orange')
    plt.plot([0, nlags +1], [-0.1, -0.1], '--', color = 'orange')

def acf_tau(x, nlags = 40):
    "nlags is how long the acf calculated series will be"
    tau = np.where(acf(x, nlags) > abs(0.1))[0][-1] # largest index with autocorrelation above treshold abs(0.1)
    tau += 1
    return tau

def e_folding_delay(x, nlags = 40, non_neg_acf = True):
    """
    Autocorrelation time 1/e for finding suitable delay
e    
    p.27 State Space Reconstruction pdf by Ng Sook Kien
    """
    if non_neg_acf == True:
        values = abs(abs(acf(rolling[50], nlags = 40))-(1/np.exp(1)))
    else:
        values = abs(acf(rolling[50], nlags = 40)-(1/np.exp(1)))
    min_dist = min(values)
    tau = np.where(values == min_dist)[0][0]
    return tau


tau_array = np.zeros(len(rolling))
iter_i = 0
start_time_main = time.time()
for i in rolling:
    tau_array[iter_i] = e_folding_delay(i, non_neg_acf = False)
    iter_i += 1
print("--- %s seconds ---" % (time.time() - start_time_main))

"delay is always 1 if we use autocorrelation abs(1/e), otherwise 2"

plot_acf(rolling[1])
plot_acf(rolling[100])
plot_acf(rolling[200])
plot_acf(rolling[600])
plot_acf(rolling[4000])   
plt.xlabel('Time lag')
plt.ylabel('Autocorrelation')
plt.title('ACF')
 
#False nearest neighbors to calculate optimal embedding dimension

window_size = 2000
emb_dim = 65 # FNN_param50 indicated 65 is best embedding dimension
gap = window_size
rolling = rolling_window(df.logR_ask.dropna(), window_size, gap) #dropped NaN from logR

pypsr.global_false_nearest_neighbors(sw_1, tau)
FNN_param = pypsr.global_false_nearest_neighbors(sw_1, tau, max_dims = 40)
FNN_param3 = pypsr.global_false_nearest_neighbors(rolling[3], tau, max_dims = 40)
FNN_param50 = pypsr.global_false_nearest_neighbors(rolling[50], tau, max_dims = 40)
FNN_param50_1 = pypsr.global_false_nearest_neighbors(rolling[50], tau, max_dims = 50)
FNN_param50_2 = pypsr.global_false_nearest_neighbors(rolling[50], tau, max_dims = 70)
FNN_param50_2 = pypsr.global_false_nearest_neighbors(rolling[50], tau, max_dims = 100)

#sample 5 FNN
FNN_param50_1 = pypsr.global_false_nearest_neighbors(rolling[1], tau, max_dims = 200)
FNN_param50_2 = pypsr.global_false_nearest_neighbors(rolling[100], tau, max_dims = 200)
FNN_param50_3 = pypsr.global_false_nearest_neighbors(rolling[200], tau, max_dims = 200)
FNN_param50_4 = pypsr.global_false_nearest_neighbors(rolling[600], tau, max_dims = 200)
FNN_param50_5 = pypsr.global_false_nearest_neighbors(rolling[4000], tau, max_dims = 200)

plt.plot(FNN_param50_1[1])
plt.plot(FNN_param50_2[1])
plt.plot(FNN_param50_3[1])
plt.plot(FNN_param50_4[1])
plt.plot(FNN_param50_5[1])
plt.xlabel('Embedding dimension')
plt.ylabel('% False neighbours')
plt.title('False nearest neighbours of random windows')



tau = 1
loop_range = np.arange(0,4100,100)
FNN_array = np.zeros(len(loop_range))
start_time_main = time.time()
iter_i = 0
for i in loop_range:
    FNN = pypsr.global_false_nearest_neighbors(rolling[i], tau, max_dims = 200)
    min_dist = min(FNN[1])
    emb_dim = np.where(FNN[1] == min_dist)[0][0]
    FNN_array[iter_i] = emb_dim
    iter_i += 1
print("--- %s seconds ---" % (time.time() - start_time_main))
    
# Derivative less than 0.002

# Check FNN sparsely, very heavy computing time
tau = 1
loop_range = np.arange(0,4100,100)
FNN_array = np.zeros(len(loop_range))  #emd-dim based on lowest FNN
dFNN_array = np.zeros(len(loop_range)) #emd-dim based on when dFNN lower than treshold 0.002
start_time_main = time.time()
iter_i = 0
for i in loop_range:
    FNN = pypsr.global_false_nearest_neighbors(rolling[i], tau, max_dims = 200)
    min_dist = min(FNN[1])
    emb_dim = np.where(FNN[1] == min_dist)[0][0]
    FNN_array[iter_i] = emb_dim
    
    for j in range(len(FNN[1])):
        try:
            if FNN[1][j] - FNN[1][j+1] < 0.002:
                dFNN_array[iter_i] = j #If dFNN still 0 --> it is acutally NaN < use this marginal gains after
                break
        except:
            dFNN_array[iter_i] = np.NaN
    iter_i += 1
    print("Done: " + str(float(i)/float(loop_range[-1]))+" Time Elapsed %s s " % (time.time() - start_time_main))
print("--- %s seconds ---" % (time.time() - start_time_main))    

#np.savetxt("W2000GW_logRask_FNN_loop0-4100-100_tau1_maxdim200_standardized.csv", FNN_array, delimiter = ',')
#np.savetxt("W2000GW_logRask_dFNN_loop0-4100-100_tau1_maxdim200_standardized.csv", dFNN_array, delimiter = ',')
FNN_r = np.genfromtxt("W2000GW_logRask_FNN_loop0-4100-100_tau1_maxdim200_standardized.csv")
dFNN_r = np.genfromtxt("W2000GW_logRask_dFNN_loop0-4100-100_tau1_maxdim200_standardized.csv")
plt.plot(dFNN_r), plt.legend(["dFNN < 0.002"])

plt.plot(FNN_r), plt.legend(["min FNN(0-200)"])
plt.xlabel('Window item')
plt.ylabel('Embedding dimension')
plt.title('FNN')


plt.plot(dFNN_r), plt.plot([0, 40], [np.mean(dFNN_r), np.mean(dFNN_r)], '--'), plt.legend(["dFNN < 0.002", "mean"])
plt.xlabel('Window item')
plt.ylabel('Embedding dimension')
plt.title('dFNN')

############################################## 
    
"ENTROPY CALCULATION LOOP"
window_size = 2000
emb_dim = 35 # FNN_param50 indicated 65 is best embedding dimension
gap = window_size
#rolling = rolling_window(df_std.logR_ask.dropna(), window_size, gap) #dropped NaN from logR
rolling = rolling_window(df_QN_laplace_std.values.transpose()[0], window_size, gap) #dropped NaN from logR
#rolling_ns = rolling_window(df.ask, window_size, 10)
#rolling_ts = rolling_window(df.index, window_size, 400)

res_array = np.zeros(len(rolling))
res_array_1 = np.zeros(len(rolling))
res_array_2 = np.zeros(len(rolling))
res_array_3 = np.zeros(len(rolling))

iter_i = 0
start_time_main = time.time()
for i in rolling: #when rolling[0] same error as vectorized maybe it goies trhough each element not row
    #res_array[iter_i] = ent.sample_entropy(i, 25)
    #res_array_1[iter_i] = ent.permutation_entropy(i, m = 4, delay =1)
    try:
        #res_array[iter_i] = ent.permutation_entropy(i, m=4, delay = 100)
        #res_array[iter_i] = nolds.sampen(i, emb_dim= emb_dim)
        res_array_1[iter_i] = ent.shannon_entropy(i)
        #res_array_2[iter_i] = nolds.lyap_r(i,emb_dim = emb_dim)
        #res_array_3[iter_i] = nolds.hurst_rs(i)
        res_array_2[iter_i] = gzip_compress_ratio(i)
        res_array_3[iter_i] = lempel_ziv_complexity(i)
    except:
        res_array[iter_i] = np.NaN
        #res_array[iter_i] = np.NaN
        res_array_1[iter_i] = np.NaN
        res_array_2[iter_i] = np.NaN
        res_array_3[iter_i] = np.NaN
    iter_i += 1
    if (iter_i % 40) == 0:
        print(float(iter_i) / float(len(rolling)))
print("--- %s seconds ---" % (time.time() - start_time_main))
    
#res_array_2 => nold.sampen(i, emb_dim = 3)
#res_array_3[iter_i] = nolds.corr_dim(i, emb_dim=emb_dim)

################################################

####
#READ IN LANDSCAPE ARRAY

####
"""
ls_file_name = [
        "W2000GW_Landscape_AlphaShape_dim1_KK1_tseq3l300.csv",
        "W2000GW_Landscape_AlphaComplex10_embdim35_tau1.csv",
        "W2000GW_Landscape_QN_AlphaComplex_dim1_KK1_tseq10l300.csv",
        "W2000GW_Landscape_AlphaComplex_dim1_KK1_tseq20l300.csv",
        "W2000GW_Landscape_AlphaComplex_dim2_KK1_tseq20l300.csv",
        "W2000GW_Landscape_QN_AlphaComplex_dim2_KK1_tseq10l300.csv",
        "W1000GW_Landscape_AlphaShape_dim1_KK1_tseq20l300_Trunc_i5612.csv",
        ]
"""
ls_file_name = listdir(path)
"""
ls_file_name = [
        "std_W2000GW_Landscape_AlphaShape_dim1_KK1_tseq10l300.csv",
        "std_W2000GW_QN_Landscape_AlphaShape_dim1_KK1_tseq10l300.csv",
        "std_W2000GW_QN_Landscape_AlphaShape_dim1_KK1_tseq10l300_discrete.csv",
        "std_W2000GW_QN_Landscape_AlphaShape_dim1_KK1_tseq10l300_discrete77.csv",
        "std_ma1000_W2000GW_Landscape_AlphaShape_emddim35_tau1.csv"
        ]
"""
ls1 = pd.read_csv(ls_file_name[3])
ls2 = pd.read_csv(ls_file_name[6])
ls = pd.read_csv(ls_file_name[3])
# sample ls
#sample_size = int(len(ls.columns) / 8)
#ls = ls.sample(sample_size, axis = 1)


tseq = np.arange(0,10,float(10)/float(300))
tseq = tseq+tseq[1]
plt.plot(tseq , ls[ls.columns[10]].values)

ls['index'] = tseq
ls = ls.set_index('index')
#MEAN landscape
#ls.transpose().mean().plot()
#ls.transpose().std().plot()

ls_mean = ls.transpose().mean() #mean of rows i.e. x values
ls_std = (ls.transpose().std()) #std of rows i.e x values
ls['mean'] = ls_mean
ls['std'] = ls_std

# make empirical confidence bound, bound values [0, infy]
#ls_upper = (ls['mean'] + 1.96*ls['std'])
#ls_lower = (ls['mean'] - 1.96*ls['std'])
#ls_lower[ls_lower < 0 ] = 0
#ls['mean'].plot()
#ls_upper.plot()
#ls_lower.plot()

# make empirical confidence bound with positive skew - Recaluclate from #MEAN landscape
ls_upper = (ls['mean'] + 1.96*ls['std'])
ls_lower = (ls['mean'] - 1.96*ls['std'])
ls_lower_neg = (ls['mean'] - 1.96*ls['std'])
ls_upper_unskew = (ls['mean'] + 1.96*ls['std'])
ls_max = ls.transpose().max()
ls_min = ls.transpose().min()
ls_neg_index = ls_lower[ls_lower < 0 ].index
ls_upper[ls_neg_index] = ls_upper[ls_neg_index] + abs(ls_lower[ls_lower < 0])
ls_lower[ls_lower < 0] = 0

plt.figure(figsize = (8,6), dpi = 80,)
plt.gca().set_ylim([0,0.3])
ls['mean'].plot(legend = True)
#ls['mean'].plot(legend = True)
ls_upper_unskew.name = "Upper 95 %" #"Upper 95 % unskewed"
ls_upper_unskew.plot(style = ':', legend = True)
ls_lower.name = "Lower 95 % bounded"
ls_lower.plot(style = ':', legend = True)
plt.title("EURUSD mean landscape")

ls_upper.name = "Upper 95 % skewed"
ls_upper.plot(style = '--', legend = True)
ls_lower_neg.name = "Lower 95 & unbounded"
ls_lower_neg.plot(style = ':', legend = True)
ls_max.name = "maximum persistence"
ls_max.plot(legend = True)
ls_min.name = "minimum persistence"
ls_min.plot(legend = True)

#for i in range(1):
#    ls[ls.columns[i]].plot()


#mean landscape dp distributions
ls.transpose().mean().plot()
ls.transpose().std().plot()

#for i in ls.transpose().columns:
from scipy.stats import norm
std = np.std(ALIV_adj)
mean = np.mean(ALIV_adj)

h = sorted(ALIV_adj.values)
h = pd.DataFrame(h)
fit = norm.pdf(h, mean, std)
plt.plot(h,fit, '-')
plt.hist(h, normed = True, bins = 100)
plt.title("ALIV distribution")   


#std = ls.transpose().std()
#mean = ls.transpose().mean()
X_row = 10
mean = ls.transpose().mean()
std = ls.transpose().std()
h = sorted(ls.transpose()[ls.transpose().columns[X_row]])
h = pd.DataFrame(h)
#h = h[h > 0].dropna()
fit = norm.pdf(h, mean[X_row], std[X_row])
plt.plot(h, fit, '-')
plt.hist(h, normed = True, bins = 100)
plt.title("dist")
plt.boxplot([ls.transpose()[ls.transpose().columns[1]], ls.transpose()[ls.transpose().columns[2]]])
plt.boxplot([ls.transpose()[ls.transpose().columns[i]] for i in range(len(ls))])
# We must look at each row not X1, X2, etx. X1[1], X2[1] - is the correct

"ENTROPY CALCULATION LOOP"


#COMPLEXITY CALCULATION LOOP FOR LS
ls_complexity_array = np.zeros(len(ls.columns))
iter_i = 0
start_time_main = time.time()
for i in range(len(ls.columns)): #when rolling[0] same error as vectorized maybe it goies trhough each element not row
    try:
        ls_complexity_array[i] = ent.shannon_entropy(ls[ls.columns[i]])
        #ls_complexity_array[i] = nolds.sampen(sw_1, emb_dim= emb_dim)
    except:
        ls_complexity_array[i] = np.NaN
    #iter_i += 1
    if (i % 40) == 0:
        print(float(i) / float(len(ls.columns)))
print("--- %s seconds ---" % (time.time() - start_time_main))


#LOOKING AT PERSISTENT LANDSCAPE TO DETERMINE NOISYNESS BY LOOKING AT AMOUNT OF FEATS ABOVE TRESHOLD e

ls_persistent_array = np.zeros(len(ls.columns))
ls_persistentSum_array = np.zeros(len(ls.columns))
ls_persistentMax_array = np.zeros(len(ls.columns))
ls_persistentMax_array_treshold = np.zeros(len(ls.columns))
eps = 0.0 # max(ls_persistentMax_array of QN) + 0.02 = 0.35 eps = 0.35 DEFAULT
start_time_main = time.time()
for i in range(len(ls.columns)):
    temp_ls = ls[ls.columns[i]]
    persistent_count = len(temp_ls[temp_ls > eps]) # counting number of points above treshold
    persistent_sum = sum(temp_ls[temp_ls > eps]) # integral of points above treshold <--- possibly a good measure
    persistent_max = max(temp_ls)
    max_ls_tresholded = temp_ls[temp_ls > eps]
    if np.size(max_ls_tresholded) == 0: max_ls_tresholded = 0 
    else: max_ls_tresholded = max(temp_ls[temp_ls > eps])
    persistent_max_tresh = max_ls_tresholded # remove all noise maxes
    ls_persistent_array[i] = persistent_count
    ls_persistentSum_array[i] = persistent_sum
    ls_persistentMax_array[i] = persistent_max
    ls_persistentMax_array_treshold[i] = persistent_max_tresh
    if (i % 40) == 0:
        print(float(i) / float(len(ls.columns)))
print("--- %s seconds ---" % (time.time() - start_time_main))

#test if all ls_persistentMax_array_treshold larger than eps
min(ls_persistentMax_array_treshold[ls_persistentMax_array_treshold > 0])

plt.figure(figsize = (8,6)), plt.gca().set_ylim([10, 210]),plt.plot(ls_persistent_array)
plt.title("Laplace QN Persistence integrals")

############### Distribution of persistent integrals

ls_persistent_integrals = np.zeros(len(ls.columns))
#ls_persistent_integrals = np.zeros(len(range(2000,4000)))

start_time_main = time.time()
for i in range(len(ls.columns)):
#for i in range(2000,4000):
    temp_ls = ls[ls.columns[i]]
    persistent_integral = temp_ls.sum()
    #ls_persistent_integrals[i-2000] = persistent_integral #must adjust with start value
    ls_persistent_integrals[i] = persistent_integral
    if (i % 40) == 0:
        print(float(i)  / float(len(ls.columns)))
print("--- %s seconds ---" % (time.time() - start_time_main))


from scipy.stats import probplot
from scipy import stats
import pylab
probplot(ls_persistent_integrals, dist=stats.norm, plot=pylab)
probplot(ls_persistent_integrals, dist=stats.rayleigh, sparams = (0.5, std), plot=pylab)
#plt.title("Normal QQ-plot")

x = stats.loggamma.rvs(c=2.5, size = 500)
res = stats.probplot(x, dist=stats.loggamma, sparams=(2.5,), plot = pylab)


from scipy.stats import norm
from scipy.stats import rayleigh
from scipy.stats import laplace

mean = ls_persistent_integrals.mean()
#mean = 4.7
std = np.sqrt(ls_persistent_integrals.var())

plt.figure(figsize = (8,6))
#plt.gca().set_ylim(0, 0.16)
h = sorted(ls_persistent_integrals)
h = pd.DataFrame(h)
fit = norm.pdf(h, mean, std)
#fit_2 = rayleigh.pdf(h, 1.4, 4)
#fit_2 = rayleigh.pdf(h, 1 ,0.5)
plt.plot(h,fit, '-', label = "N(1.6, 6)"), plt.legend()
plt.hist(h, normed = True, bins = 100, label = "Empirical dist")#, plt.legend()
#plt.plot(h,fit_2, '-', label = "Rayleigh(loc = 1, scale = 0.5)"), plt.legend()
plt.title("QN Empirical persistent integral distribution")   



############### TEST: mean integral - integral
"""
ls_persistent_diffs = np.zeros(len(ls_persistent_integrals))
for i in range(len(ls_persistent_integrals)):
    temp = mean - ls_persistent_integrals[i]
    ls_persistent_diffs[i] = temp

from scipy.stats import norm

mean = ls_persistent_diffs.mean()
std = np.sqrt(ls_persistent_diffs.var())

h = sorted(ls_persistent_diffs)
h = pd.DataFrame(h)
fit = norm.pdf(h, mean, std)
plt.plot(h,fit, '-')
plt.hist(h, normed = True, bins = 100)
plt.title("Persistent integral distribution")   
"""

#############################
""" Clustering on landscapes """
#############################
"""
ls_file_name = [
        "W2000GW_Landscape_AlphaShape_dim1_KK1_tseq3l300.csv",
        "W2000GW_Landscape_AlphaComplex10_embdim35_tau1.csv",
        "W2000GW_Landscape_QN_AlphaComplex_dim1_KK1_tseq10l300.csv",
        "W2000GW_Landscape_AlphaComplex_dim1_KK1_tseq20l300.csv",
        "W2000GW_Landscape_AlphaComplex_dim2_KK1_tseq20l300.csv",
        "W2000GW_Landscape_QN_AlphaComplex_dim2_KK1_tseq10l300.csv",
        ]
"""
"""
ls_file_name = [
        "std_W2000GW_Landscape_AlphaShape_dim1_KK1_tseq10l300.csv",
        "std_W2000GW_QN_Landscape_AlphaShape_dim1_KK1_tseq10l300.csv",
        "std_W2000GW_QN_Landscape_AlphaShape_dim1_KK1_tseq10l300_discrete.csv",
        "std_W2000GW_QN_Landscape_AlphaShape_dim1_KK1_tseq10l300_discrete77.csv",
        "std_ma1000_W2000GW_Landscape_AlphaShape_emddim35_tau1.csv"
        ]
"""
ls_file_name = listdir(path)

ls_EURUSD = pd.read_csv(ls_file_name[0])
ls_QN = pd.read_csv(ls_file_name[2])
ls_QN.columns = ls_QN.columns+"_QN"
ls_concat = pd.concat([ls_EURUSD, ls_QN], axis = 1)


def sample_Landscape_dendrogram(details = True, method = "single"):
    if details == True:
        sample_index_labels = ["X1", "X2", "X3", "X4", "X5",
                               "X6", "X7", "X8", "X9", 
                               "QN1", "QN2", "QN3", "QN4"]
    else:
        sample_index_labels = ["X", "X", "X", "X", "X",
                               "X", "X", "X", "X", 
                               "QN", "QN", "QN", "QN"]
    # sample df will be transposed in relation to ls_concat
    sample_df = pd.DataFrame([ls_concat["X1"],
                  ls_concat["X2"],
                  ls_concat["X100"],
                  ls_concat["X200"],
                  ls_concat["X500"],
                  ls_concat["X750"],
                  ls_concat["X751"],
                  ls_concat["X1000"],
                  ls_concat["X2200"],
                  ls_concat["X1_QN"],
                  ls_concat["X10_QN"],
                  ls_concat["X20_QN"],
                  ls_concat["X30_QN"]], index = sample_index_labels)
    
    #sample_df = sample_df.transpose()
    pdist = sp.spatial.distance.pdist(sample_df)
    Z = sp.cluster.hierarchy.linkage(pdist, method = method)
    dendrogram = sp.cluster.hierarchy.dendrogram(Z, labels = sample_index_labels)

sample_Landscape_dendrogram(details = False, method = "complete")

"CREATE LABELS"
"transposed relative to samples 1-5"
boolean = sample_df.index.str.endswith("QN")
np.place(sample_df.index.values, boolean*1 == 1, ["QN"])
np.place(sample_df.index.values, boolean*1 == 0, ["X"])

def reduce_labels_clust(df_samp, end_str):
    boolean = df_samp.columns.str.endswith(end_str)
    np.place(df_samp.columns.values, boolean, [end_str])

A_start, A_end = 0, 10 
B_start, B_end = 15, 20
C_start, C_end = 100, 120
D_start, D_end = 200, 240
E_start, E_end = 300, 320
QN_start, QN_end = 0, 90

"""
A_start, A_end = 0, 200
B_start, B_end = 200, 400
C_start, C_end = 400, 600
D_start, D_end = 600, 800
E_start, E_end = 800, 1000
QN_start, QN_end = 0, 90
"""

sample1 = ls_EURUSD[ls_EURUSD.columns[A_start:A_end]]
sample1.columns = sample1.columns +"_A"
reduce_labels_clust(sample1, "A")
sample2 = ls_EURUSD[ls_EURUSD.columns[B_start:B_end]]
sample2.columns = sample2.columns + "_B"
reduce_labels_clust(sample2, "B")
sample3 = ls_EURUSD[ls_EURUSD.columns[C_start:C_end]]
sample3.columns = sample3.columns + "_C"
reduce_labels_clust(sample3, "C")
sample4 = ls_EURUSD[ls_EURUSD.columns[D_start:D_end]]
sample4.columns = sample4.columns + "_D"
reduce_labels_clust(sample4, "D")
sample5 = ls_EURUSD[ls_EURUSD.columns[E_start:E_end]]
sample5.columns = sample5.columns + "_E"
reduce_labels_clust(sample5, "E")
sampleQN = ls_QN[ls_QN.columns[QN_start:QN_end]]
sampleQN.columns = sampleQN.columns + "_QN"
reduce_labels_clust(sampleQN, "QN")

sample_df = pd.concat([sample1, sample2, sample3, sample4, sample5, sampleQN], axis = 1)
sample_df = sample_df.transpose()
pdist = sp.spatial.distance.pdist(sample_df)
Z = sp.cluster.hierarchy.linkage(pdist, method = "complete")
plt.figure(figsize = (25,10))
plt.title("Hierarchal Clustering Dendrogram")
plt.xlabel("sample index")
plt.ylabel("Distance")
dendrogram = sp.cluster.hierarchy.dendrogram(Z, labels = sample_df.index)


""" plot SW after generating signal.gaussian pulse
plt.plot(t, np.sin(t_pi)+i+e), plt.axvline(x = [-1], color = 'red', linestyle = '--', ymax = .75), plt.axvline(x = -.5, color = 'red', linestyle = '--', ymax = .75), plt.axhline(y=1.35, color = 'red', linestyle = '--', xmin = 0.045, xmax = .27), plt.text(-0.8,1.5, r'$M_\tau$', fontsize = 15), plt.text(-0.97,-1.12, r't', fontsize = 11), plt.text(-.47, -1.12, r'$t+M_\tau$', fontsize = 11), plt.text(-0.87, 1.1, "window", fontsize = 11)
"""

#Normalizing complexity df
from sklearn import preprocessing
comp_df = pd.read_csv("std_W2000GW_logRask_complexity.csv")
comp_df = comp_df.fillna(0)

# adjusting infinite value of sampen35  to largest value of sampen35 column
max_sampen35 = comp_df[np.isfinite(comp_df.sampen35) == True].sampen35.max()
inf_loc_sampen35 = comp_df[np.isinf(comp_df.sampen35) == True].sampen35.index
comp_df.sampen35[inf_loc_sampen35] = 0 #max_sampen35

x = comp_df.values
#normalization
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
norm_comp_df = pd.DataFrame(x_scaled, columns = comp_df.columns)

#standardization
scaler = preprocessing.StandardScaler().fit(x)
rescaledX = scaler.transform(x)
std_comp_df = pd.DataFrame(rescaledX, columns = comp_df.columns)

#find max value index
comp_df[comp_df.shannon == comp_df.shannon.max()]
comp_df[comp_df.gzip == comp_df.gzip.max()]

# discretize laplace dist
(df_QN_laplace_std.head()*100000).round()/100000
((df_QN_laplace_std.head()*100000).round()/100000).values.transpose()[0]
#DISCRETIZE BELOW
pd.qcut(((df_QN_laplace_std.head()*100000).round()/100000).values.transpose()[0], 4, labels = False)


"""
CHECK HOW MANY DISCRETE VALUES DUE TO TRUNCATIONS WHEN PLOTTING
https://stackoverflow.com/questions/41665659/count-of-unique-value-in-column-pandas
"""

unique_values_df = df.logR_ask.dropna().value_counts()
unique_df_discrete_df = pd.DataFrame((unique_values_df.index*1000000))
unique_df_discrete_df = unique_df_discrete_df.round()
unique_df_discrete_df = pd.Series(unique_df_discrete_df.values.transpose()[0])
uniques = unique_df_discrete_df.value_counts()
number_of_uniques = len(uniques)

# make QN laplace get 100 different outcomes only because 77 outcomes for stock price
#QN_test = pd.qcut(((df_QN_laplace_std*100000).round()/100000).values.transpose()[0], 100, labels = False)
scaler = 4.22 #scaler = 4.22 gives 77 uniques, 1 gives discretization that looks like stocks std
df_QN_laplace_std_lim = (df_QN_laplace_std*scaler).round()/scaler
df_QN_laplace_std_lim[0:2000].plot()

unique_vals = pd.Series(df_QN_laplace_std_lim.values.transpose()[0]).value_counts()
num_of_uniques = len(unique_vals)


####3
df_std_roll1000 = df_std.rolling(1000).sum()/1000

# complexitt plots
#com_arr = pd.read_csv("EURUSD complexity")
#QN_com_arr = pd.read_csv("QN complexity")
plt.figure(figsize=(8,6), dpi = 80), 
plt.gca().set_ylim(-10, 10),
((QN_com_arr.gzip - QN_com_arr.gzip.mean())/QN_com_arr.gzip.std()).plot(legend = True), 
((QN_com_arr.shannon-QN_com_arr.shannon.mean())/QN_com_arr.shannon.std()).plot(alpha = 0.8, legend = True), 
plt.title("Laplace QN Standardized complexity")

QN_ent = pd.DataFrame()
QN_ent["PersistentIntegral"] = (ls_persistent_array - ls_persistent_array.mean())/ ls_persistent_array.std() #QN in per array
QN_ent["Gzip"] = (QN_com_arr.gzip - QN_com_arr.gzip.mean())/QN_com_arr.gzip.std()
QN_ent["Shannon"] = (QN_com_arr.shannon-QN_com_arr.shannon.mean())/QN_com_arr.shannon.std()


plt.figure(figsize=(8,6), dpi = 80), 
plt.plot((ls_persistent_array - ls_persistent_array.mean())/ls_persistent_array.std())
((com_arr.gzip - com_arr.gzip.mean())/com_arr.gzip.std()).plot(alpha = 0.8, legend = True), 
((com_arr.shannon-com_arr.shannon.mean())/com_arr.shannon.std()).plot(alpha = 0.5, legend = True),

EURUSD_ent = pd.DataFrame()
EURUSD_ent["PersistentIntegral"] = (ls_persistent_array - ls_persistent_array.mean())/ls_persistent_array.std()
EURUSD_ent["Gzip"] = (com_arr.gzip - com_arr.gzip.mean())/com_arr.gzip.std()
EURUSD_ent["Shannon"] = (com_arr.shannon-com_arr.shannon.mean())/com_arr.shannon.std()
#Correlation of series
std_EURUSD_gzip = (com_arr.gzip - com_arr.gzip.mean())/com_arr.gzip.std()
std_EURUSD_shan = (com_arr.shannon-com_arr.shannon.mean())/com_arr.shannon.std()
EURUSD_corr = std_EURUSD_gzip.corr(std_EURUSD_shan)

std_QN_gzip = (QN_com_arr.gzip - QN_com_arr.gzip.mean())/QN_com_arr.gzip.std()
std_QN_shan = (QN_com_arr.shannon-QN_com_arr.shannon.mean())/QN_com_arr.shannon.std()
QN_corr = std_QN_gzip.corr(std_QN_shan)


plt.figure(figsize= (8,6)), plt.gca().set_ylim(-6, 10),EURUSD_ent.PersistentIntegral.plot(label = "Persistence", legend = True), EURUSD_ent.Gzip.plot(alpha = 0.8, legend = True), EURUSD_ent.Shannon.plot(alpha = 0.5, legend = True), plt.title("Standardized EURUSD complexity and persistence")
plt.figure(figsize= (8,6)), plt.gca().set_ylim(-6, 10),QN_ent.PersistentIntegral.plot(label = "Persistence", legend = True), QN_ent.Gzip.plot(alpha = 0.8, legend = True), QN_ent.Shannon.plot(alpha = 0.5, legend = True), plt.title("Standardized QN complexity and persistence")
