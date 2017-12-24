# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 03:34:43 2017

@author: Patrick


Script for generating final results plots

"""



from os import listdir
from os.path import isfile, join
from os import chdir
from os import getcwd

path = "C:\\Users\\Patrick\\Documents\\MEX\\MEX\\Local laptop backup\\Other Financial Data\\TDA\\Paris_Intraday\\Final Results CSV\\Complexity"
chdir(path)
print(getcwd())


ls_file_name = listdir(path)


EURUSD_W1000 = pd.read_csv(ls_file_name[0])
EURUSD_W1000MA1000 = pd.read_csv(ls_file_name[1])
EURUSD_W2000 = pd.read_csv(ls_file_name[2])
EURUSD_W2000MA1000 = pd.read_csv(ls_file_name[3])

QN_W1000 = pd.read_csv(ls_file_name[4])
QN_W1000MA1000 = pd.read_csv(ls_file_name[5])
QN_W2000 = pd.read_csv(ls_file_name[6])
QN_W2000MA1000 = pd.read_csv(ls_file_name[7])


col = EURUSD_W1000.columns[2]


EURUSD_W1000[col].plot()
EURUSD_W1000MA1000[col].plot()
EURUSD_W2000[col].plot()
EURUSD_W2000MA1000[col].plot()
QN_W1000[col].plot()
QN_W1000MA1000[col].plot()
QN_W2000[col].plot()
QN_W2000MA1000[col].plot()

plt_list = [
EURUSD_W1000[col],
EURUSD_W1000MA1000[col],
EURUSD_W2000[col],
EURUSD_W2000MA1000[col],
QN_W1000[col],
QN_W1000MA1000[col],
QN_W2000[col],
QN_W2000MA1000[col]
]

plt_titles = [
        "EURUSD W = 1000",
        "EURUSD MA1000 W = 1000",
        "EURUSD W = 2000",
        "EURUSD MA1000 W = 2000",
        "QN W = 1000",
        "QN MA1000 W = 1000",
        "QN W = 2000",
        "QN MA1000 W = 2000"]

i = 7
plt.figure(figsize = (8,6)), 
plt.gca().set_ylim([0, 400]),
plt_list[i].plot()
plt.title(col + " " +plt_titles[i])

i = 7
ls_persistent_integrals = plt_list[i]
mean = ls_persistent_integrals.mean()
std = np.sqrt(ls_persistent_integrals.var())

plt.figure(figsize = (8,6))
h = sorted(ls_persistent_integrals)
h = pd.DataFrame(h)
fit = norm.pdf(h, mean, std)
#plt.plot(h,fit, '-', label = "N(1.6, 6)"), plt.legend()
plt.hist(h, normed = True, bins = 100, label = "Empirical dist")#, plt.legend()
#plt.plot(h,fit_2, '-', label = "Rayleigh(loc = 1, scale = 0.5)"), plt.legend()
plt.title("Empirical Dist of Persistence Integral " +plt_titles[i]) 

########################
### ALL PLOT
#########################
ls_persistent_integrals = plt_list[0]
mean = ls_persistent_integrals.mean()
std = np.sqrt(ls_persistent_integrals.var())

ls_persistent_integrals_1 = plt_list[1]
mean_1 = ls_persistent_integrals_1.mean()
std_1 = np.sqrt(ls_persistent_integrals_1.var())

ls_persistent_integrals_2 = plt_list[2]
mean_2 = ls_persistent_integrals_2.mean()
std_2 = np.sqrt(ls_persistent_integrals_2.var())

ls_persistent_integrals_3 = plt_list[3]
mean_3 = ls_persistent_integrals_3.mean()
std_3 = np.sqrt(ls_persistent_integrals_3.var())

ls_persistent_integrals_4 = plt_list[4]
mean_4 = ls_persistent_integrals_4.mean()
std_4 = np.sqrt(ls_persistent_integrals_4.var())

ls_persistent_integrals_5 = plt_list[5]
mean_5 = ls_persistent_integrals_5.mean()
std_5 = np.sqrt(ls_persistent_integrals_5.var())

ls_persistent_integrals_6 = plt_list[6]
mean_6 = ls_persistent_integrals_6.mean()
std_6 = np.sqrt(ls_persistent_integrals_6.var())

ls_persistent_integrals_7 = plt_list[7]
mean_7 = ls_persistent_integrals_7.mean()
std_7 = np.sqrt(ls_persistent_integrals_7.var())


plt.figure(figsize = (8,6))
plt.gca().set_ylim([0, 0.02]),
plt.gca().set_xlim([0, 200])

h = sorted(ls_persistent_integrals)
h = pd.DataFrame(h)
plt.hist(h, normed = True, bins = 100, label = plt_titles[0]), plt.legend()

h = sorted(ls_persistent_integrals_1)
h = pd.DataFrame(h)
plt.hist(h, normed = True, bins = 100, label = plt_titles[1]), plt.legend()

h = sorted(ls_persistent_integrals_2)
h = pd.DataFrame(h)
plt.hist(h, normed = True, bins = 100, label = plt_titles[2]), plt.legend()

h = sorted(ls_persistent_integrals_3)
h = pd.DataFrame(h)
plt.hist(h, normed = True, bins = 100, label = plt_titles[3]), plt.legend()

h = sorted(ls_persistent_integrals_4)
h = pd.DataFrame(h)
plt.hist(h, normed = True, bins = 100, label = plt_titles[4]), plt.legend()

h = sorted(ls_persistent_integrals_5)
h = pd.DataFrame(h)
plt.hist(h, normed = True, bins = 100, label = plt_titles[5]), plt.legend()

h = sorted(ls_persistent_integrals_6)
h = pd.DataFrame(h)
plt.hist(h, normed = True, bins = 100, label = plt_titles[6]), plt.legend()

h = sorted(ls_persistent_integrals_7)
h = pd.DataFrame(h)
plt.hist(h, normed = True, bins = 100, label = plt_titles[7]), plt.legend()

#plt.plot(h,fit_2, '-', label = "Rayleigh(loc = 1, scale = 0.5)"), plt.legend()
plt.title("Empirical Distributions of Persistence Integrals")   
