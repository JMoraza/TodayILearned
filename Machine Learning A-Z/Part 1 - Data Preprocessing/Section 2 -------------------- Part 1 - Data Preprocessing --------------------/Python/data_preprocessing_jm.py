#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 20:50:01 2020

@author: joannamoraza
"""

#Importing libraries
import numpy as np  
#Imports numpy which allows for working with arrays
import pandas as pd 
#Imports pandas, library that allows working w/ matrix & dataframes
import matplotlib.pyplot as plt  
#Imports matplot library which allows working w/ graphs & visuals

#Importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:3].values  #iloc.[:,:-1] is another way of indexing
y = dataset.iloc[:,3]  #iloc.[:,-1] is another way of indexing

print(x)
print(y)

