# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 12:43:15 2017

@author: Patrick
"""
import pandas as pd
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
from os import chdir
from os import getcwd

chdir("C:\\Users\\Patrick\\Documents\\data\\green")
print(getcwd())

greens_list = []
for f in range(len(listdir())):
    if (listdir()[f][0:5] == "green") and (listdir()[f][-4:] == '.csv'):
        greens_list.append(listdir()[f])
    
# Get header of first file to get the correct headers
# Read only header line to save processing time
#header_df = pd.read_csv(greens_list[0], nrows = 0)

""" METHODS TO CONVERT ERRONEOUS (SHORT) HEADERS TO CORRECT (LONG) VERSION
###
Some csv files have headers with PO_Loc and DO_Loc instead of 
PO_Latidude, PO_Longtitude, DO_Latitude and DO_Longtitude. 

We want the long version, so we import a file with the long version and
replace the header of the short version with the long version. This is 
because we want the rest of the data cells not to be shifted because of
PO_Loc and DO_Loc are a cell with (Latitude,Longtitude). This makes readin
wrong.

###
                                 
# One method
dft = pd.read_csv(greens_list[6], skiprows = [0], header = None)
dft.columns = list(header_df)

# Second method
dft = pd.read_csv(greens_list[6], skiprows = [0], header = None)
dft = pd.DataFrame(dft, columns = list(header_df))
"""


for f in greens_list:
    
    cols = ['Trips', 'Fare amount (avg)', 'Fare amount (std)',
               'Tip amount (avg)', 'Tip amount (std)', 'Total amount (avg)', 'Total amount (std)']
    d_ts = pd.DataFrame(data = None, index = None, columns = cols)
        
    # Readin summary
    if "summary_ts.csv" in listdir():
        d_ts_in = pd.read_csv("summary_ts.csv", index_col = 0)#, parse_dates=[0], infer_datetime_format = True, index_col = 0)
    else:
        d_ts_in = pd.DataFrame(data = None, index = None, columns = cols) 
    print(f)
    print("OK")
    # Readin with index in first column and Date and Time separete columns 
    #df = pd.read_csv(f, parse_dates=[0], infer_datetime_format = True)
    
    # Read in csv, do not read in header
    df = pd.read_csv(f, skiprows = [0], header = None)
    header_df = pd.read_csv(f, nrows = 0)
    
    # Change header to our universal header version
    #try:
    #    df.columns = list(header_df)
    #except:
        # Rename column by column - probabily there is better/faster method for this
    for j in range(len(list(header_df))):
        df.rename(columns={j:list(header_df)[j].capitalize()}, inplace = True)
        #print(list(header_df)[j])
        
    
    temp = pd.DatetimeIndex(df['Lpep_pickup_datetime']) #Base Date Time on lpep_pickup_datetime
    df['Date'] = temp.date
    df['Time'] = temp.time
    
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
        
    # Append indata and current data
    d_ts = d_ts_in.append(d_ts)
    
    # Generate new file to build upon
    d_ts.to_csv('summary_ts.csv', sep = ',')
    
"""        
for f in range(len(listdir())):
    if (listdir()[f][0:5] == "green") and (listdir()[f][-4:] == '.csv'):
        
        cols = ['Trips', 'Fare amount (avg)', 'Fare amount (std)',
               'Tip amount (avg)', 'Tip amount (std)', 'Total amount (avg)', 'Total amount (std)']
        d_ts = pd.DataFrame(data = None, index = None, columns = cols)
        
        # Readin summary
        if "summary_ts.csv" in listdir():
            d_ts_in = pd.read_csv("summary_ts.csv", index_col = 0)#, parse_dates=[0], infer_datetime_format = True, index_col = 0)
        else:
            d_ts_in = pd.DataFrame(data = None, index = None, columns = cols) 
        print(listdir()[f])
        print("OK")
        # Readin with index in first column and Date and Time separete columns 
        df = pd.read_csv(listdir()[f], parse_dates=[0], infer_datetime_format = True)
        temp = pd.DatetimeIndex(df['lpep_pickup_datetime']) #Base Date Time on lpep_pickup_datetime
        df['Date'] = temp.date
        df['Time'] = temp.time
        
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
        
        # Append indata and current data
        d_ts = d_ts_in.append(d_ts)
        
        # Generate new file to build upon
        d_ts.to_csv('summary_ts.csv', sep = ',')
"""

""" ToDo
Go through all files in location
if columns (headers) are stated in PO_location and DO_location 
           Remake to PO_longtitude, PO_Latitude, DO_longtitude and DO_latitude

"""

#Prototype code for changing stuff

#TODO
"""
Still wrong with the readin
"""
