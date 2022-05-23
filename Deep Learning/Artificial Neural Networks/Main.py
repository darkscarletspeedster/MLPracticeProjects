# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

print(tf.__version__) # helps to see version of any library


## 1. Data Preprocessing
# Importing the dataset
dataset = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Deep Learning\Artificial Neural Networks\Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values # starting columns are irrelevant
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

# One Hot Encoding the 'Geography column'
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling (Compulsory in NN and applied to all features)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


## 2. Building the ANN
# Initializing the ANN
ann = tf.keras.models.Sequential() # initialises ANN as sequence of layers

# Adding the input and the first hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu')) # units is number of nodes, here 6 which is based on trial and test
    # relu = rectifier activation function

# Adding the second hidden layer (as going for deeper networking)
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu')) # parameters can be changed for better accuracy

# Adding the output layer
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid')) # if more than 2 categories as output then more than 1 output neuron required
    # along with One hot encoding, here output is just 1 or 0
    # sigmoid will give us probability for the output as well
    # when categorical output then use 'softmax' instead of 'sigmoid'


## 3. Training the ANN
# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # optimizer is Stochastic Gradient Descent, here Adam Optimizer
    # loss function is used for calculating difference between actual and predicted result
    # for binary results as in this case 'binary_crossentropy' is used always
    # for non-binary 'categorical_crossentropy' can be used
    # metrics is used for evalution of the model using different parameters on training set, here only 'accuracy' is used

# Training the ANN on the Training set
ann.fit(x_train, y_train, batch_size = 32, epochs = 100) # 32 is a classic value usually used for batch size
    # epochs cannot be too small as it requires it for better learning


## 4. Making the predictions and evaluating the model
# Predicting the result of a single observation
print('Single Prediction: ', ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))

# Predecting the Test set results
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5) # converts into true and false i.e. 1 and 0
#print('Test Results:', np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:', cm)
print('Accuracy score:', accuracy_score(y_test, y_pred) * 100, '%')