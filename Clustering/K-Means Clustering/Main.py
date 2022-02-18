# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Clustering\K-Means Clustering\Mall_Customers.csv')
x = dataset.iloc[:, [3, 4]].values # first column has customer Ids which is irrelavent and only last two
    # columns are considered so that K-means can be visualized easily on a 2-D graph

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
# we run K-Means 10 times for trying clusters 1 to 10
# then WCSS will be calculated
wcss = []
for i in range(1, 11) :
    kmeans = KMeans(n_clusters = i, random_state = 0, init = "k-means++")
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss) # plotting the graph for wcss
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

# training the K-means model on the dataset
kmeans = KMeans(n_clusters = 5, random_state = 0, init = "k-means++")
y = kmeans.fit_predict(x) # fit predict would provide us with 5 groups (clusters) i.e. the dependent variable
print(y) #naming conventions for the clusters might differ but actual result remains the same

# visualizing the clusters, 5 scatters for 5 clusters
plt.scatter(x[y == 0, 0], x[y == 0, 1], s = 100, c = 'red', label = 'C1') # will select all rows where y = 0
plt.scatter(x[y == 1, 0], x[y == 1, 1], s = 100, c = 'blue', label = 'C2')
plt.scatter(x[y == 2, 0], x[y == 2, 1], s = 100, c = 'green', label = 'C3')
plt.scatter(x[y == 3, 0], x[y == 3, 1], s = 100, c = 'cyan', label = 'C4')
plt.scatter(x[y == 4, 0], x[y == 4, 1], s = 100, c = 'purple', label = 'C5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title("Clusters of Customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()