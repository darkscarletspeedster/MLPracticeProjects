# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Model Selection and Boosting\Model Selection\Social_Network_Ads.csv')
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

# Training the Kernel SVM model
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:', cm)
print('Accuracy score:', accuracy_score(y_test, y_pred) * 100, '%')

# Applying k-Flod Cross Validation
# This is done to better evaluate the perfomance of the model
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10) 
    # normally folds selected are 10 and folds are set of randomly selected test set
print("Mean Accurancy: {:.2f} %".format(accuracies.mean() * 100)) # {} gives a formatting during printing, here with 2 decimal places
    # format function takes in the value that needs to be formated
    # mean() is a function of lists which will provide mean of all different accuracies
print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100)) # calulate standard deviation 
    # it is done to check the variance between the accuracies and if they are close or far from each other
    # more than 4 is a bit higher stating that accuracies lie between 86-94 if mean is 90

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']}, # contains different combinations of super parameters we intent to tune in
        # c is used for reducing overfitting, less the value of c, better the regualrization 
    {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], # 2 dictionares as Gamma function is used only in rbf kernel, it is the coefficient of its formula
        'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy',
    cv = 10, n_jobs = -1) # n_jobs is optional and using -1 says that use all processors as this is a heavy process
grid.fit(x_train, y_train) # training for all combinations of hyper paramters
best_accuracy = grid.best_score_
best_params = grid.best_params_
print("Best Accurancy: {:.2f} %".format(best_accuracy * 100))
print("Best Paramters: ", best_params)

# Visualising the Training set results
# Code only for training purposes and not real life situation as real life sit  uation would have more than 2 features where it isn't used
from matplotlib.colors import ListedColormap
x_set, y_set = sc.inverse_transform(x_train), y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 10, stop = x_set[:, 0].max() + 10, step = 0.25), 
                     np.arange(start = x_set[:, 1].min() - 1000, stop = x_set[:, 1].max() + 1000, step = 0.25))
                     # increasing step decreases time but makes the curve less smoother
plt.contourf(x1, x2, classifier.predict(sc.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape), alpha = 0.75,
             cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)) : 
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM Intuition (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
x_set, y_set = sc.inverse_transform(x_test), y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 10, stop = x_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = x_set[:, 1].min() - 1000, stop = x_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(x1, x2, classifier.predict(sc.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape), alpha = 0.75,
             cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)) : 
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM Intuition (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()