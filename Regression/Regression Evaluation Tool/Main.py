# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# Generic Settings
dataset = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Regression\Regression Evaluation Tool\Data.csv')
display_outcome = False # shows actual ouput vs predicted output
np.set_printoptions(precision=2) # sets property for output arrays
class color: # class for print in particular format
   CYAN = '\033[96m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# Allocating x and y paramters from the dataset
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

########################################## Multiple Linear Regression
print(color.BOLD + color.RED + 'Multiple Linear Regression Start -------------------' + color.END)

# Training
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x_train, y_train)

# Predict
linear_y_pred = linear_regressor.predict(x_test)

# Print if asked
if display_outcome :
    print(np.concatenate((linear_y_pred.reshape(len(linear_y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
print ('R2 Score:', r2_score(y_test, linear_y_pred))

print(color.BOLD + color.RED + 'Multiple Linear Regression End ---------------------' + color.END)
##########################################
########################################## Polynomial Regression
print(color.BOLD + color.BLUE + 'Polynomial Regression Start ------------------------' + color.END)

# Training
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x_train)
poly_regressor = LinearRegression()
poly_regressor.fit(x_poly, y_train)

# Predict
poly_y_pred = poly_regressor.predict(poly_reg.transform(x_test))

# Print if asked
if display_outcome :
    print(np.concatenate((poly_y_pred.reshape(len(poly_y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
print ('R2 Score:', r2_score(y_test, poly_y_pred))

print(color.BOLD + color.BLUE + 'Polynomial Regression End --------------------------' + color.END)
##########################################
########################################## Decision Tree Regression
print(color.BOLD + color.CYAN + 'Decision Tree Regression Start ---------------------' + color.END)

# Training
from sklearn.tree import DecisionTreeRegressor
decision_tree_regressor = DecisionTreeRegressor(random_state = 0)
decision_tree_regressor.fit(x_train, y_train)

# Predict
decision_tree_y_pred = decision_tree_regressor.predict(x_test)

# Print if asked
if display_outcome :
    print(np.concatenate((decision_tree_y_pred.reshape(len(decision_tree_y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
print ('R2 Score:', r2_score(y_test, decision_tree_y_pred))

print(color.BOLD + color.CYAN + 'Decision Tree Regression End -----------------------' + color.END)
##########################################
########################################## Random Forest Regression
print(color.BOLD + color.GREEN + 'Random Forest Regression Start ---------------------' + color.END)

# Training
from sklearn.ensemble import RandomForestRegressor
random_forest_regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
random_forest_regressor.fit(x_train, y_train)

# Predict
random_forest_pred = random_forest_regressor.predict(x_test)

# Print if asked
if display_outcome :
    print(np.concatenate((random_forest_pred.reshape(len(random_forest_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
print ('R2 Score:', r2_score(y_test, random_forest_pred))

print(color.BOLD + color.GREEN + 'Random Forest Regression End -----------------------' + color.END)
##########################################
########################################## Support Vector Regression
print(color.BOLD + color.YELLOW + 'Support Vector Regression Start --------------------' + color.END)

# Feature Scaling
y_train = y_train.reshape(len(y_train),1)
y_test = y_test.reshape(len(y_test),1)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train = sc_x.fit_transform(x_train)
y_train = sc_y.fit_transform(y_train)

# Training
from sklearn.svm import SVR
svr_regressor = SVR(kernel = 'rbf')
svr_regressor.fit(x_train, y_train.ravel()) # ravel will convert array shape tp (n, ) (i.e. flatten it) 1D array

# Predict
svr_y_pred = sc_y.inverse_transform(svr_regressor.predict(sc_x.transform(x_test)))

# Print if asked
if display_outcome :
    print(np.concatenate((svr_y_pred.reshape(len(svr_y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
print ('R2 Score:', r2_score(y_test, svr_y_pred))

print(color.BOLD + color.YELLOW + 'Support Vector Regression End ----------------------' + color.END)
##########################################