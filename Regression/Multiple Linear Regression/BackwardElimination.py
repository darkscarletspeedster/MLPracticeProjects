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

x = x[:, 1:] # avoiding dummy variable trap by removing first row as one hot encoding puts new columns at the beginning

x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1) # in axis 1 is adding column and 0 is adding a row
# this is done in order to give x0 of MLR equation a value as b0 is constant cause x0 is 1
# but in-order for our model to understand that, we are adding a column of 1's
# arr given 1's Column so that it is added in the beginning of the matrix and if replace
# position of x with arr value, 1's column will be added in the end

# splitting dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Building the optimal model using Backward Elimination manually
import statsmodels.api as sm

x_opt = np.array(x_train[:, [0, 1, 2, 3, 4, 5]], dtype=float) # taking all variables initially
regressor = sm.OLS(endog = y_train, exog = x_opt).fit()
# endog is dependent variable, exog is feature variables which does not include intercept
# and because of this we added that 1's column

#print(regressor.summary()) # provides useful information about MLR model

# x2 has highest P value 99% > 5% so we remove that column by repeating last three code lines
x_opt = np.array(x_train[:, [0, 1, 3, 4, 5]], dtype=float) # taking all variables initially
regressor = sm.OLS(endog = y_train, exog = x_opt).fit()
#print(regressor.summary()) 

# x1 has highest P value 94% > 5% so we remove that column by repeating last three code lines
x_opt = np.array(x_train[:, [0, 3, 4, 5]], dtype=float) # taking all variables initially
regressor = sm.OLS(endog = y_train, exog = x_opt).fit()
#print(regressor.summary()) 

# x2 has highest P value 60% > 5% so we remove that column by repeating last three code lines
x_opt = np.array(x_train[:, [0, 3, 5]], dtype=float) # taking all variables initially
regressor = sm.OLS(endog = y_train, exog = x_opt).fit()
#print(regressor.summary()) 
#x1 shows P value 0 but P value can neber be 0 it's just too small

# x2 has highest P value 6% > 5% so we remove that column by repeating last three code lines
x_opt = np.array(x_train[:, [0, 3]], dtype=float) # taking all variables initially
regressor = sm.OLS(endog = y_train, exog = x_opt).fit()
#print(regressor.summary()) 
# x3 the R&D spend has the highest statistical significance in predicting profit

# predicting
x_test = x_test[:, [0, 3]]
y_pred = np.array(regressor.predict(x_test), dtype = float)
np.set_printoptions(precision = 2) # limits to 2 decimal places
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))