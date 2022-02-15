# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# Generic Settings
dataset = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Classification\Classification Evaluation Tool\Data.csv')
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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

########################################## Logistic Regression
print(color.BOLD + color.RED + 'Logistic Regression Start -------------------' + color.END)

# Training
from sklearn.linear_model import LogisticRegression
logistic_classifier = LogisticRegression(random_state = 0)
logistic_classifier.fit(x_train, y_train)

# Predict
logistic_y_pred = logistic_classifier.predict(x_test)

# Print if asked
if display_outcome :
    print('Test Results:', np.concatenate((logistic_y_pred.reshape(len(logistic_y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, logistic_y_pred)
print('Confusion Matrix:', cm)
print('Accuracy score:', accuracy_score(y_test, logistic_y_pred) * 100, '%')

print(color.BOLD + color.RED + 'Logistic Regression End ---------------------' + color.END)
##########################################
########################################## K-NN Classifier
print(color.BOLD + color.BLUE + 'K-NN Classifier Start -------------------' + color.END)

# Training
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) # default parameters
knn_classifier.fit(x_train, y_train)

# Predict
knn_y_pred = knn_classifier.predict(x_test)

# Print if asked
if display_outcome :
    print('Test Results:', np.concatenate((knn_y_pred.reshape(len(knn_y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, knn_y_pred)
print('Confusion Matrix:', cm)
print('Accuracy score:', accuracy_score(y_test, knn_y_pred) * 100, '%')

print(color.BOLD + color.BLUE + 'K-NN Classifier End ---------------------' + color.END)
##########################################
########################################## SVM Classifier
print(color.BOLD + color.CYAN + 'SVM Classifier Start -------------------' + color.END)

# Training
from sklearn.svm import SVC
svm_classifier = SVC(kernel = 'linear', random_state = 0)
svm_classifier.fit(x_train, y_train)

# Predict
svm_y_pred = svm_classifier.predict(x_test)

# Print if asked
if display_outcome :
    print('Test Results:', np.concatenate((svm_y_pred.reshape(len(svm_y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, svm_y_pred)
print('Confusion Matrix:', cm)
print('Accuracy score:', accuracy_score(y_test, svm_y_pred) * 100, '%')

print(color.BOLD + color.CYAN + 'SVM Classifier End ---------------------' + color.END)
##########################################
########################################## Kernel SVM Classifier
print(color.BOLD + color.CYAN + 'Kernel SVM Classifier Start -------------------' + color.END)

# Training
from sklearn.svm import SVC
ksvm_classifier = SVC(kernel = 'rbf', random_state = 0)
ksvm_classifier.fit(x_train, y_train)

# Predict
ksvm_y_pred = ksvm_classifier.predict(x_test)

# Print if asked
if display_outcome :
    print('Test Results:', np.concatenate((ksvm_y_pred.reshape(len(ksvm_y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, ksvm_y_pred)
print('Confusion Matrix:', cm)
print('Accuracy score:', accuracy_score(y_test, ksvm_y_pred) * 100, '%')

print(color.BOLD + color.CYAN + 'Kernel SVM Classifier End ---------------------' + color.END)
##########################################
########################################## Naive Bayes Classifier
print(color.BOLD + color.GREEN + 'Naive Bayes Classifier Start -------------------' + color.END)

# Training
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)

# Predict
nb_y_pred = nb_classifier.predict(x_test)

# Print if asked
if display_outcome :
    print('Test Results:', np.concatenate((nb_y_pred.reshape(len(nb_y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, nb_y_pred)
print('Confusion Matrix:', cm)
print('Accuracy score:', accuracy_score(y_test, nb_y_pred) * 100, '%')

print(color.BOLD + color.GREEN + 'Naive Bayes Classifier End ---------------------' + color.END)
##########################################
########################################## Decision Tree Classifier
print(color.BOLD + color.YELLOW + 'Decision Tree Classifier Start -------------------' + color.END)

# Training
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt_classifier.fit(x_train, y_train)

# Predict
dt_y_pred = dt_classifier.predict(x_test)

# Print if asked
if display_outcome :
    print('Test Results:', np.concatenate((dt_y_pred.reshape(len(dt_y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, dt_y_pred)
print('Confusion Matrix:', cm)
print('Accuracy score:', accuracy_score(y_test, dt_y_pred) * 100, '%')

print(color.BOLD + color.YELLOW + 'Decision Tree Classifier End ---------------------' + color.END)
##########################################
########################################## Random Forest Classifier
print(color.BOLD + color.YELLOW + 'Random Forest Classifier Start -------------------' + color.END)

# Training
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier.fit(x_train, y_train)

# Predict
rf_y_pred = rf_classifier.predict(x_test)

# Print if asked
if display_outcome :
    print('Test Results:', np.concatenate((rf_y_pred.reshape(len(rf_y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, rf_y_pred)
print('Confusion Matrix:', cm)
print('Accuracy score:', accuracy_score(y_test, rf_y_pred) * 100, '%')

print(color.BOLD + color.YELLOW + 'Random Forest Classifier End ---------------------' + color.END)
##########################################