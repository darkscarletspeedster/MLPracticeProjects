# Making a hybrid deep learning model
# Importing the libraries
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Part 1: Identify the Frauds with the Self-Organizing Map
# Importing the dataset
dataset = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Bonus\Mega Case Study For ANN and SMO\Credit_Card_Applications.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
x = sc.fit_transform(x)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1, learning_rate = 0.5) 
som.random_weights_init(x)
som.train_random(data = x, num_iteration = 100)

# Filtering the Frauds
dist = som.distance_map()
cordinates = []
for i in range(0, 10):
    arr = dist[i]
    arr = np.where(arr >= 0.9)
    if len(arr[0]) != 0:
        for j in arr:
            cordinates.append((i, j[0]))

mappings = som.win_map(x)
list = []
for i in cordinates:
    inner_list = mappings[i]
    for j in inner_list:
        list.append(j)

frauds = sc.inverse_transform(list)[:, 0]


## Part 2: Going from Unsupervised to Supervised Deep Learning
# Creating the matrix of features
customers = dataset.iloc[:, 1:].values

# Creating he dependent vairable 1 - Fraud and 0 - No Fraud
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input and the first hidden layer
classifier.add(Dense(units = 2, activation = 'relu', kernel_initializer = 'uniform', input_dim = 15)) # input_dim = number of features

# Adding the output layer
classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2) # few observations only so less number of batch size and epochs

# Predicting the probablities of frauds
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[(-y_pred[:, 1]).argsort()]
np.set_printoptions(suppress = True)
print("Predictions: ", y_pred)