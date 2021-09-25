# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Regression\Multiple Linear Regression\50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

# splitting dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# traning the multiple linear regression model on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting results
y_pred = regressor.predict(x_test)
np.set_printoptions(precision = 2) # limits to 2 decimal places
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)) #reshaping makes print vertically instead of horizontally reshape(rows, columns)
# 1 in second parameter of concatenate means vertical concatenation

# predicting single outcome
print('Single prediction: ', regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

# Coefficient and Intercept
print('Coefficients: ', regressor.coef_)
print('intercept: ', regressor.intercept_)
