 # Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Reinforcement Learning\Upper Confidence Bound (UCB)\Ads_CTR_Optimisation.csv')

# Implementing UCB
import math # for calculation square root

N = 10000
D = 10
ads_selected = []
numbers_of_selections = [0] * D # initialises the list of 10 0's
sums_of_rewards = [0] * D
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0

    for d in range(0, D):
        if (numbers_of_selections[d] > 0):
            average_reward = sums_of_rewards[d] / numbers_of_selections[d]
            delta_d = math.sqrt((3 * math.log(n + 1)) / (2 * numbers_of_selections[d]))
            upper_bound = average_reward + delta_d
        else:
            upper_bound = 1e400 # gives a high value # does upper bound calculates more than this?

        if (upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = d # updating index of selected of ad
            # after 10 plus rounds all the ads would be selected at least once

    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward

# Visualizing the results
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()