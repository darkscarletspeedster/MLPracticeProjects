# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Model Selection and Boosting\XGBoost\Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Training the XGBoost
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train) # new version requires this extra steps
    # need to understand this as giving wrong accuracy score
y_test = le.transform(y_test)

from xgboost import XGBClassifier # XGBRegressor gives us regression model
classifier = XGBClassifier()
classifier.fit(x_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:', cm)
print('Accuracy score:', accuracy_score(y_test, y_pred) * 100, '%')

# Applying k-Flod Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10) 
print("Mean Accurancy: {:.2f} %".format(accuracies.mean() * 100)) 
print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))