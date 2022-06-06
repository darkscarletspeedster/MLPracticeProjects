## Part 1: Data Preprocessing
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the training set
dataset_train = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Deep Learning\Recursive Neural Networks\Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2:].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
# we are applying normalization here and not standardization as better suited to RNNs especially containing sigmoid function in its hidden layer
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
# this special data structure is used to store previous states of RNN
# 60 (can be flexible) timesteps means it will look at previous 60 stock values at every new state in time and predict one output
# here as each entry corresponds to new day, 60 corresponds to roughly 3 months of previous data
x_train = [] # contains prev 60 data for every entry
y_train = [] # contains next day stock price
for i in range (60, 1258): # starts with 60 as goes back 60 so can't start at 0, 1258 comes from dataset (should be more robust method)
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0]) # for first loop 59 is t, here we take i = 60 which is t + 1

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping
# more than one parameters might be influential for the output, so adding more dimensions to dataset
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# within brackets 1st is batch_size = which is the length training set here
# 2nd is timesteps
# 3rd is number of indicators
# we have here just changed the shape of the array not added new features, but we should do this before adding new indicators


## Part 2: Building the RNN
# Importing libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout # used for regularization and remove over fitting

# Intialising the RNN
# Pytorch over keras should be preferred normally as it is powerful tool
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
# units = 50(high as more trends need to recorded) = no. of cells(neurons)
# return_sequences = true, as more than one LSTM layers
# input_shape = shape of our array we created above, 3 parameters but 3rd is taken care off
regressor.add(Dropout(0.2)) # corresponds to 20 % of ignoring the LSTM layer neurons

# Adding the second LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True)) # input_shape is inherited through previous layers
regressor.add(Dropout(0.2))

# Adding the third LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the fourth LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50)) # return_sequence is removed as no more LSTM layers after this and by default value is false
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') # we use adam although keras website says some other optimizer
    # mean_squared_error as we are doing regression

# Fitting the RNN to the Training Set
# Basically training the model
regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)
# loss does not go below more than what we get because of the dropout we added, but if it goes then it is overfitting


## Part 3: Making the predictions and visualizing the results
# Getting the real stock prize of 2017
dataset_test = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Deep Learning\Recursive Neural Networks\Google_Stock_Price_Test.csv')
test_set = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0) # 0 is for vertical and 1 for horizontal
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values # gives lower limit which is wrong way to compute
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
x_test = []
for i in range (60, 80): 
    x_test.append(inputs[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_pred = regressor.predict(x_test)
y_pred = sc.inverse_transform(y_pred)

# Visaulising the results
plt.plot(test_set, color = 'red', label = 'Real Goggle Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Goggle Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()