# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Clustering\Hierarchical Clustering\Mall_Customers.csv')
x = dataset.iloc[:, [3, 4]].values # first column has customer Ids which is irrelavent and only last two
    # columns are considered so that Hierarchical clustering can be visualized easily on a 2-D graph

# Using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x, method = 'ward')) # ward minimises variance b/w the points so it very usefull in creating 
    # a better dendograms

plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distances") # always on y-axis of dendogram
plt.show()

# it shows that there are 2 optimal number of clusters 3 and 5 but as in K-means the elbow method showed as it was 5
# we move ahead with 5 number of clusters, but this shows that a dataset might have more than 1 optimal amounts of clusters

# training the Hierarchical model on the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
y = hc.fit_predict(x) # fit predict would provide us with 5 groups (clusters) i.e. the dependent variable
print(y) #naming conventions for the clusters might differ but actual result remains the same

# visualizing the clusters, 5 scatters for 5 clusters
plt.scatter(x[y == 0, 0], x[y == 0, 1], s = 100, c = 'red', label = 'C1') # will select all rows where y = 0
plt.scatter(x[y == 1, 0], x[y == 1, 1], s = 100, c = 'blue', label = 'C2')
plt.scatter(x[y == 2, 0], x[y == 2, 1], s = 100, c = 'green', label = 'C3')
# tried with 3 clusters as dendogram suggests the 3 won't be bad
# although the the output graph shows 5 would be better
plt.scatter(x[y == 3, 0], x[y == 3, 1], s = 100, c = 'cyan', label = 'C4')
plt.scatter(x[y == 4, 0], x[y == 4, 1], s = 100, c = 'purple', label = 'C5')
plt.title("Clusters of Customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()