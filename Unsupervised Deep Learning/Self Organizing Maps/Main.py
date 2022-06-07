# Importing the libraries
from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Unsupervised Deep Learning\Self Organizing Maps\Credit_Card_Applications.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values # only to give customers who were approved but were not dependable

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
x = sc.fit_transform(x)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1, learning_rate = 0.5) 
    # x and y can be aything, here the data is small so grid size 10 x 10
    # input_len is the number of columns including customer Ids
    # sigma is used for the convergence radius, here kept to default value 1
    # learning_rate is used for updating the weights, higher the rate faster it is, here default 0.5
    # decay_function can be used for improving the convergence
som.random_weights_init(x)
som.train_random(data = x, num_iteration = 100) # iterations can be any

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show, subplots, scatter

fig, ax = subplots()
bone()
pcolor(som.distance_map().T) # T is for taking transpose
colorbar() #(orientation = 'horizontal')
markers = ['o', 's']
colors = ['r', 'c']

for i, j in enumerate(x):
    w = som.winner(j)
    plot(w[0] + 0.5, w[1] + 0.5, # cordinates of winning node, 0.5 is added to put the marker at the center
        markers[y[i]], # selects symbols for customers who got selected or not
        markeredgecolor = colors[y[i]], # gives color to the marker selected above
        markerfacecolor = 'None',
        markersize = 10,
        markeredgewidth = 2
        )

mappings = som.win_map(x)
annot = ax.annotate("", xy=(0,0), xytext=(20, 20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"))
annot.set_visible(False)

def update_annot(list):
    text = "{}".format("\n".join([str(elem) for elem in list]))
    annot.set_text(text)


def hover(event):
    vis = annot.get_visible()
    x = int(event.xdata)
    y = int(event.ydata)
    inner_list = mappings[(x, y)]
    if inner_list:
        list = []
        for k in inner_list:
            list.append(k)

        list = sc.inverse_transform(list)[:, 0]
        annot.xy = (x, y)
        annot.set_visible(True)
        update_annot(list)
        fig.canvas.draw_idle()
    else:
        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

def click(event):
    x = int(event.xdata)
    y = int(event.ydata)
    inner_list = mappings[(x, y)]
    if inner_list:
        list = []
        for k in inner_list:
            list.append(k)

        list = sc.inverse_transform(list)[:, 0]
        print("Clicked: ", list)

fig.canvas.mpl_connect("motion_notify_event", hover)
fig.canvas.mpl_connect("button_press_event", click)

show()