# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Simple Linear Regression\Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Training SLR on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the test set result
y_pred = regressor.predict(x_test)

# visualizing traing set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visualizing test set results
plt.scatter(x_test, y_test, color = 'red')
plt.scatter(x_test, y_pred, color = 'green')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.plot(x_test, y_pred, color = 'yellow')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# making a single predict, example of an employee with 12 years of experience
print('Predicted value for 12 years of Experience: ', regressor.predict([[12]])[0])

# getting final linear regression equation with the values of the coefficients
print('Coefficient: ', regressor.coef_[0])
print('Intercept: ', regressor.intercept_)