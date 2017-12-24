# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt

#df = pd.DataFrame.from_csv("green_tripdata_2016-01.csv")
#df = pd.DataFrame.from_csv("trip_1.csv")
#headers = list(df.columns.values)

# Readin with index in first column and Date and Time separete columns 
df = pd.read_csv("green_tripdata_2016-01.csv", parse_dates=[0], infer_datetime_format = True)
temp = pd.DatetimeIndex(df['lpep_pickup_datetime']) #Base Date Time on lpep_pickup_datetime
df['Date'] = temp.date
df['Time'] = temp.time

print(nfp)
# Unique dates

print(df['Date'].unique())
# first and last pickup and dropoff of dataset
print("First pickup: "+str(df[headers[0]].min()))
print("First dropoff: "+str(df[headers[1]].min()))
print("Last pickup: "+str(df[headers[0]].max()))
print("Last dropoff: "+str(df[headers[1]].max()))

# Histogram of passenger count
#df['Passenger_count'].hist(cumulative=True, normed = 1, bins = 100)
df['Passenger_count'].plot(kind = 'hist', normed = 1, title = 'Passenger count')
#df['Trip_distance'].hist(cumulative=True, normed = 1, bins = 100)

# Trip Distance
df['Trip_distance'].plot(kind = 'hist', normed = 1, title = 'Trip distrance', bins = 100)
df[df['Trip_distance'] <= 20]['Trip_distance'].plot(kind = 'hist', normed = 1, title = 'Trip distrance', bins = 100)

# Tip
df['Tip_amount'].plot(kind = 'hist', normed = 1, title = 'Tip amount')
df[(df['Tip_amount'] <= 20) & (df['Tip_amount'] > 0)]['Tip_amount'].plot(kind = 'hist', normed = 1, title = 'Tip amount', bins = 100)
df[(df['Tip_amount'] <= 20) & (df['Tip_amount'] > 0) & (df['Payment_type'] == 1)]['Tip_amount'].plot(kind = 'hist', normed = 1, title = 'Tip amount (Credit Card)', bins = 100)
df[(df['Tip_amount'] <= 20) & (df['Tip_amount'] > 0) & (df['Payment_type'] == 2)]['Tip_amount'].plot(kind = 'hist', normed = 1, title = 'Tip amount (Cash)', bins = 100)

"All tippers tip with Credit Card"

df.plot.hexbin(x = [1,2], y = [2,3])

"Port daily relevant statistics of each month to csv time series"
"Perform some initial correlation analysis and stuff"

"If destination is relevant then we need to learn to use hexbin and narrow down Point-of-Interests"
"Perhaps do not need to learn, but enough with df indexing"
"Hexbin for visualization"

""" Green Cab
N-trips per day
Fare-amount
Tip amount
Total amount
Trip type
"""

""" Yellow Cab
N-trips per day
Fare-amount
Tip amount
Total amount
Trip type
"""

"""
We index for daily values
We calculate std dev and mean for daily data
"""

#Index for daily values change df['Date'].unique()[1]] part
"df[df['Date'] == df['Date'].unique()[1]]"