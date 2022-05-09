 # Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Reinforcement Learning\Thompson Sampling Algorithm\Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
import random # for picking up random distributions

N = 10000
D = 10
ads_selected = []
numbers_of_rewards_1 = [0] * D # no of reward 1 up to round n
numbers_of_rewards_0 = [0] * D # no of reward 0 up to round n 
total_reward = 0

for n in range(0, N):
    ad = 0
    max_random = 0

    for d in range(0, D):
        random_beta = random.betavariate(numbers_of_rewards_1[d] + 1, numbers_of_rewards_0[d] + 1)
        if (random_beta > max_random):
            max_random = random_beta
            ad = d
    
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1
    
    total_reward += 1
    

# Visualizing the results
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()