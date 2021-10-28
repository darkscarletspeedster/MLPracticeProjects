# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Regression\Decision Tree Regression\Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# training the Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0) # fixes the seed and provides consistent results
regressor.fit(x, y)

# predicting a new result
print('Prediction: ', regressor.predict([[6.5]]))

# visualising the DTR results(high resolution only, as this example of 2-D dataset isn't fit for DTR 
# and would still be providing a poor visual)
x_grid = np.arange(min(x), max(x), 0.1) # makes graph plotting to 0.1 instead of 1
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show() # result is predictive as the model just splits on every x value
# thus low resolution graph would have made no sense as it would have simply given the same output as input