# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 12:21:23 2017

@author: Patrick
"""


print("green_tripdata_2016-01")
print("green_tripdata_2016-02")

from os import listdir
from os.path import isfile, join

# We can have as listdir()[i] in for instead
if listdir()[1][0:5] == "green":
    print("ok")
    
if listdir()[1][-4:] == '.csv':
    print("ok")
    
for f in range(len(listdir())):
    if (listdir()[f][0:5] == "green") and (listdir()[f][-4:] == '.csv'):
        print(listdir()[f])