#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:22:54 2021

@author: joannamoraza
"""

#Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Data
##Note, not importing the 1st column with job titles because the 2nd column
##represents the level of each job position.

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

print(x)
print(y) 

#Training the Simple Linear Regression Model on the Whole Dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#Creating the Polynomial Linear Regression Model
from sklearn.preprocessing import PolynomialFeatures
poly_feat = PolynomialFeatures(degree = 4) #Degree is the power of x
x_poly = poly_feat.fit_transform(x)        #Created new feature matrix
##Note, the degree was 2 but was updated to 4 for a better curve fit.

#Training the Polynomail Linear Regression Model w/ New Feature Matrix
poly_lin_reg = LinearRegression()
poly_lin_reg.fit(x_poly,y)

#Visualizing the Simple Linear Regression Model
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Truth or Bluff (Simple Linear Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#Visualizing the Polynomial Linear Regression Model
plt.scatter(x,y,color='red')
plt.plot(x,poly_lin_reg.predict(x_poly),color='blue')
plt.title('Polynomial Linear Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#Visualizing the Polynomial Linear Regression Model (with higher resolution & smoother curve)
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, poly_lin_reg.predict(poly_feat.fit_transform(X_grid)), color = 'blue')
plt.title('Smoother Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Predicting a Salary Using Linear Regression Model
##x = 6.5
print(lin_reg.predict([[6.5]]))

#Predicting a Salary Using Polynomial Regression Model
##x = 6.5
print(poly_lin_reg.predict(poly_feat.fit_transform([[6.5]])))

