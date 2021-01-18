#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 22:37:55 2021

@author: joannamoraza
"""

#Importing libraries

import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt  


#Importing the dataset

dataset = pd.read_csv('Salary_Data.csv')
##Years of experience is independent variable x, salary is dependent variable y 
x = dataset.iloc[:,:1].values
y = dataset.iloc[:,1].values

print(x)
print(y)


#Splitting the training set and test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

#Note, no need for feature scaling since there is only one feature x.

#Training the Simple Regression Model with the training set
from sklearn.linear_model import LinearRegression
LR = LinearRegression()  ##This is the model
LR.fit(x_train, y_train) ##Here you are training the model w/ your data
##Regression is the prediction of an actual value.

#Predicting the test results
y_pred = LR.predict(x_test)
print(y_pred)

#Visualizing Training Set
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, LR.predict(x_train), color = 'blue')
plt.title('Experience vs. Salary (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing Testing Set
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, LR.predict(x_train), color = 'blue')
plt.title('Experience vs. Salary (Testing Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()