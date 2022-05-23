# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf


## 1. Data Preprocessing
# Importing the dataset
dataset = pd.read_excel(r'D:\Learning\Udemy ML\MLPracticeProjects\Deep Learning\ANN for Regression\Data.xlsx')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


## 2. Building the ANN
# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input and the first hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units = 1)) # default activation function is used for regression i.e. none


## 3. Training the ANN
# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Training the ANN on the Training set
ann.fit(x_train, y_train, batch_size = 32, epochs = 100)


## 4. Making the predictions and evaluating the model
# Predecting the Test set results
y_pred = ann.predict(x_test)
np.set_printoptions(precision = 2)
print('Test Results:', np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))