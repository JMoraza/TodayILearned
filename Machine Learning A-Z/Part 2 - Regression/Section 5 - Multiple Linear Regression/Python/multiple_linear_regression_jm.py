#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 19:39:09 2021

@author: joannamoraza
"""

#Importing libraries

import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt  


#Importing the dataset

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

print(x)
print(y)


##Encoding the independent variables

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
CT = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x = np.array(CT.fit_transform(x))
print(x)


#Splitting the training set and test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

##Note, feature scaling is not necessary with Multiple Regression b/c every variable 
##coefficient will compensate for differences in range and put everything on same scale.




