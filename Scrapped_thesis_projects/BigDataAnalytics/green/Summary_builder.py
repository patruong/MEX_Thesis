# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 22:48:53 2017

@author: Patrick
"""
import pandas as pd
import matplotlib.pyplot as plt


# Readin summary
d_ts_ = pd.read_csv("summary_ts.csv", index_col = 0)#, parse_dates=[0], infer_datetime_format = True, index_col = 0)

# Readin with index in first column and Date and Time separete columns 
df = pd.read_csv("green_tripdata_2016-02.csv", parse_dates=[0], infer_datetime_format = True)
temp = pd.DatetimeIndex(df['lpep_pickup_datetime']) #Base Date Time on lpep_pickup_datetime
df['Date'] = temp.date
df['Time'] = temp.time


cols = ['Trips', 'Fare amount (avg)', 'Fare amount (std)',
       'Tip amount (avg)', 'Tip amount (std)', 'Total amount (avg)', 'Total amount (std)']
d_ts = pd.DataFrame(data = None, index = None, columns = cols)

#Go through each date
for i in df['Date'].unique():

    
    # Create mean and std for relevant variables for day i
    ntrips = len(df[df['Date'] == i])
    Fare_amount = df[df['Date'] == i]['Fare_amount'].mean()
    Fare_amount_std = df[df['Date'] == i]['Fare_amount'].std()
    Tip_amount = df[df['Date'] == i]['Tip_amount'].mean()
    Tip_amount_std = df[df['Date'] == i]['Tip_amount'].std()
    Total_amount = df[df['Date'] == i]['Total_amount'].mean()
    Total_amount_std = df[df['Date'] == i]['Total_amount'].std()
    
    temp = pd.DataFrame([[ntrips, Fare_amount, Fare_amount_std, 
                          Tip_amount, Tip_amount_std, 
                          Total_amount, Total_amount_std]], index = [i], columns = cols)
    
    d_ts = d_ts.append(temp)

#Export to csv
#d_ts.to_csv('summary_ts.csv', sep = ',')