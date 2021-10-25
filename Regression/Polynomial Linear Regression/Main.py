# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Regression\Polynomial Linear Regression\Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values #first column excluded as second column is encoded based on first one
y = dataset.iloc[:, -1].values

# not splitting data set into training and testing as its too small

# Training SLR on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)

# Training Polynomial Regression Model
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree = 5) # final exponent n value is inserted and increased gradually starting with 2
x_poly = polyReg.fit_transform(x)

# after the x variable is converted into polynomial matrix, we create new Linear regressor
# it reads the new matrix as Multiple Linear regression type
regressor2 = LinearRegression()
regressor2.fit(x_poly, y)

# visualizing the Linear Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# visualizing the Polynomial Linear Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor2.predict(x_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# on increasing the n value the results became more accurate, at 5-6 it gets locked in

# steps of making the curve smoother instead of lines joining the points
# not much important in real life as it'll include more features
x_grid = np.arange(min(x), max(x), 0.1) # makes graph plotting to 0.1 instead of 1
x_grid = x_grid.reshape((len(x_grid), 1))
x_grid_poly = polyReg.fit_transform(x_grid)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor2.predict(x_grid_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show() # showed a weired result near and after n = 14

# predicting with a new entry using Linear Regression
print('Linear Regression Result: ', regressor.predict([[6.5]]))

# predicting with a new entry using Polynomial Linear Regression
print('Polynomial Linear Regression Result: ', regressor2.predict(polyReg.fit_transform([[6.5]])))
