import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('summary_ts.csv', index_col = 0, skiprows = [i for i in range(1,4)])
#Skip first three row because they seem off

rng = pd.date_range('2013-08-05', periods = 1245, freq = 'D')
df.index = rng #Values are same so we do not need to use df.rename(index = ...)


res = sm.tsa.seasonal_decompose(df.Trips, model = 'multi')
resplot = res.plot()
plt.show()


#plt.plot(res.trend,'-', res.seasonal*res.trend, '-', res.seasonal*res.trend*res.resid, '-')
plt.plot(res.trend, '-')
plt.plot(res.seasonal*res.trend, '-', alpha = 0.5)
plt.plot(res.seasonal*res.trend*res.resid, '-', alpha = 0.3)
plt.show()
