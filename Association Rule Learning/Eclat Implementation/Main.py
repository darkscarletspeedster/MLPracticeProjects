# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Association Rule Learning\Eclat Implementation\Market_Basket_Optimisation.csv', 
    header = None) # intimates python that there are no headers in the file

# Data Preprocessing
transactions = [] # list for aprori function to accept
for i in range (0, 7501):
    transactions.append([
        str(dataset.values[i, j]) for j in range (0, 20) # 20 as most items is 20, str convert values in string acceptable by apriori
    ])  # [] is for list of products

# Training the Eclat model on the dataset
# use 'pip install apyori', no sklearn is used
from apyori import apriori
rules = apriori(transactions = transactions, 
    min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2) # can increase length for considering more products
    # min_coinfidence and min_lift can be removed for Eclat but would be good for better associations

## visualising the results
results = list(rules)

def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))

resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])

print('Result: ')
print(resultsinDataFrame.nlargest(n = 10, columns = 'Support'))
