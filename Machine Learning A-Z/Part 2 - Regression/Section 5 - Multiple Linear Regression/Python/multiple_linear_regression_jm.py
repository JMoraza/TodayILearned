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


#Creating the Multiple Linear Regression Model and Training it 

from sklearn.linear_model import LinearRegression
MLR = LinearRegression()
MLR.fit(x_train, y_train)


#Predicting the Test Results

y_pred = MLR.predict(x_test)


#Comparing the Test Results with the Model's Predictions

np.set_printoptions(precision=2) 
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),axis=1))

#BONUS

#How to predict the profit of a startup in California with R&D Spend = $160,000 
##and Admin Spend = $130,000 and Marketing = $300,000

print(MLR.predict([[1,0,0, 160000, 130000, 300000]])) 

#How to obtain the coefficients of the model.
print(MLR.coef_)
print(MLR.intercept_)
## Profit = -0.0285*Dummy Var1 + 0.0298*Dummy Var2 - 0.0124*Dummy Var3 + 0.774*R&D Spend
## - 0.0094*Admin Spend + 0.0289*Marketing Spend + 49834(intercept)

