#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 20:50:01 2020

@author: joannamoraza
"""

#Importing libraries

import numpy as np  
##Imported numpy library which supports array [matrices and vectors] operation.

import pandas as pd 
##Imported pandas library which supports cleaning and preperation of data.

import matplotlib.pyplot as plt  
##Imported pyplot module from matplot library which supports graphs & visuals.


#Importing the dataset

##Created the variable containing the data in dataframe form.
dataset = pd.read_csv('Data.csv')

##Creating a matrix of features x (also known as the independent variables). 
##and a dependent variable vector y (the information you are trying to predict w/ data).
x = dataset.iloc[:,:3].values
y = dataset.iloc[:,-1].values
##.values in panda removes axes titles of dataframe and returns values in an array (like Numpy).

print(x)
print(y)


#Taking care of missing data
  ##Missing data could cause errors when training your model.
  ##Option, remove incomplete data if sample data is large.
  ##Alternate option, use average to fill in missing data.

##Import scikit learn library which supports data processing and model building tools.
from sklearn.impute import SimpleImputer  

##Creating an instance/object from the class SimpleImputer.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
##Using .fit method to look for missing values and compute the mean.
imputer.fit(x[:,1:3])
##Using .transform method to replace the missing values w/ the computed mean.
x[:,1:3] = imputer.transform(x[:,1:3])
##This returns original matrix x with the new averaged values populated in age & salary column.
print(x)


#Encoding categorical data
##Categorical variables must be converted to numerical variables in order to use
##in the ML algorithm - this method is called one-hot encoding. 

##Encoding the independent variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
##Note, the country column is converted into a numerical array to prevent a numerical order.
CT = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x = np.array(CT.fit_transform(x))
print(x)

##Encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
y = LE.fit_transform(y)
print(y)

#Splitting the training set and test set
##Training set consists of existing data used to train your model while the test set is used 
##for testing the accuracy of the model and obtaining new observations. 
##This is the expected format.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
##test_size is 20% of data, 80% of data becomes part of training sample.
##random_state = 1 ensures data is split randomly between training and testing samples.
print(x_train)
print(x_test)
print(y_train)
print(y_test)


#Feature scaling
##Scaling the feature variables to ensure all variables are in the same range and
##one variable does not dominate another and is neglected.
##There are two techniques, standardization and normalization
from sklearn.preprocessing import StandardScaler
SC = StandardScaler()
##Note, we don't apply StandardScalar() to country column because the identifier/array
##already has values between -3 and +3 (it would mess up identifier).
x_train[:, 3:] = SC.fit_transform(x_train[:, 3:]) 
x_test[:, 3:] = SC.transform(x_test[:, 3:])
##Note, we apply the same scalars (mean and std. dev.) from training data to test data
##to maintain relativity.
print(x_train)
print(x_test)

