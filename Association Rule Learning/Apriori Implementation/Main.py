# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Association Rule Learning\Apriori Implementation\Market_Basket_Optimisation.csv', 
    header = None) # intimates python that there are no headers in the file

# Data Preprocessing
transactions = [] # list for aprori function to accept
for i in range (0, 7501):
    transactions.append([
        str(dataset.values[i, j]) for j in range (0, 20) # 20 as most items is 20, str convert values in string acceptable by apriori
    ])  # [] is for list of products

# Training the Apriori model on the dataset
# use 'pip install apyori', no sklearn is used
from apyori import apriori
rules = apriori(transactions = transactions, 
    min_support = 0.003, # 3 * 7 / 7501, i.e we want a product to occur 3 times in each transaction
        # as these values were recorded over a week therefore 3*7 and divided by total no.of transactions as you want it appear in all
    min_confidence = 0.2, # starting with 0.8 would very high se dvided by 2, some rules created, so again divided by 2
    min_lift = 3, # usually a good value, 4, 5, 6, 7, 8, 9 are also useful, below 3 not that relevant
    min_length = 2, max_length = 2) # to have only 1 product on both left and right side, i.e. if a bought then b is also bought
    # and not a rule where a,b is bought then c is also bought or vice-versa, for which values can be 3 on either side

## visualising the results
# displaying the first results coming directly from output of the apriori function
results = list(rules)
#print('Intial Results: ', results)

# putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidence = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidence, lifts))

resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
print('Beautified Results: ') 
print(resultsinDataFrame)

# Displaying the results in sorted by descending lifts
print('Orderd List: ')
print(resultsinDataFrame.nlargest(n = 10, columns = 'Lift'))
