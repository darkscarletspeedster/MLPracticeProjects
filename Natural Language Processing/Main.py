# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Natural Language Processing\Restaurant_Reviews.tsv'
    , delimiter = '\t', quoting = 3) # tsv -> tab separated, csv -> comma separated
    # quoating = 3 ignores "" in the data inorder to avoid any errors

# Cleaning the texts
# it is to covert all letters in lower case, removing punctuations and etc.
import re
import nltk # contains dataset or words
nltk.download('stopwords') # words like 'the', 'a'
from nltk.corpus import stopwords # first line downloads them and this line import thems
from nltk.stem.porter import PorterStemmer # helps in stemming of words which is taking only root of word which explain the meaning the of the word
    # if word 'loved' is used then it'll convert it into love and make it simple as meaning of he word reamins same
    # making word matrix simpler and smaller

corpus = []
for i in range (0, 1000):
    review = re.sub('[^a-z^A-Z]', ' ', dataset['Review'][i]) # this function replaces everything given in first parameter with second parameter
    review = review.lower() # converts all text in lowercase
    words = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not') # as not is a negative word and impacts interpretation and was removed from orignial data which was wrong
    words = [ps.stem(word) for word in words if not word in set(all_stopwords)] # stemming each word out of stopwords
    review = ' '.join(words)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # after analysing we can optimise this value to limit the number of words effecting the decision
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

arrat_len = len(x[0])
print('Array Length: ', arrat_len) # helps analise the arrray which can be further reduced by removing words which do not help in decision making

# Splitting the dataset into the Training set and Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0) 

# Training the Naive Bayes model on the Training Set
# many other classifiers can be used, here were using simply Naive Bayes for training purposes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predict the test results
y_pred = classifier.predict(x_test)
print('Test Results:', np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:', cm)
print('Accuracy score:', accuracy_score(y_test, y_pred) * 100, '%')

# Single review prediction
s_review = re.sub('[^a-z^A-Z]', ' ', 'Gournment....... was fantastic.')
s_review = s_review.lower()
s_review = [s_review]
print('Transformed Review: ', s_review)
s_x = cv.transform(s_review).toarray() # no removing of stopwords and other simplification as array of 1500 words already full
    # so only the words qualified will be selected from this one review and no new column will be created
print('Single review array length: ', len(s_x[0]))
print('Single review prediction: ', classifier.predict(s_x))