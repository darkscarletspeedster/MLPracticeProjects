# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Regression\Support Vector Regression\Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# feature scaling
# we scale y as the values are too high and the model won't work
y = y.reshape(len(y), 1) # need to convert y in 2D array as the traform method expects it

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)
# we create new StandardScalar Object so that it computes new Standard Deviation and Mean for y
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# training the SVR model on the whole dataset as the dataset is small and we can put in any new test data
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') # SVR take in different kernels for non-linear SVRs, rbf = Radial Basis Function Kernel
regressor.fit(x, y[:,0]) # y is given index as we had converted it into a 2D array

# predicting a new result
print('Prediction: ', sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]]))))

# visualizing the SVR Results
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color = 'blue')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# the last point isn't in the curve unlike in the polynomial curve because the last point is a outlier
# and is the farthest from the tube(outside considered error margin), infact in this scenario
# it is better model as the polynomial model was a little overfitted

# making high resolution curve
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1) # makes graph plotting to 0.1 instead of 1
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))), color = 'blue')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
