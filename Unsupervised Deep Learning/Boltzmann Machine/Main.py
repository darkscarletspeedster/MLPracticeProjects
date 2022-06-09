# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Unsupervised Deep Learning\ml-1m\movies.dat',
    sep = '::',  # sep - separator
    header = None, # as no column headers present
    engine = 'python', # better optimisation while processing the data
    encoding = 'latin-1') # contains special characters and default UTF-8 encoder would break

users = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Unsupervised Deep Learning\ml-1m\users.dat',
    sep = '::',
    header = None,
    engine = 'python',
    encoding = 'latin-1')

ratings = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Unsupervised Deep Learning\ml-1m\ratings.dat',
    sep = '::',
    header = None,
    engine = 'python',
    encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Unsupervised Deep Learning\ml-100k\u1.base',
    delimiter = '\t') # separator is tab
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Unsupervised Deep Learning\ml-100k\u1.test',
    delimiter = '\t') # separator is tab
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
# this is done to convert training_set and test_set into matrices
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0]))) # usable for different splits of training and test sets
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Converting the data into an array for training_set and test_set
# both will have same number of rows and columns where u = rows = users
# i = columns = movies and their intersection will hold movie ratings by a user for a movie
# 0 if the user hasn't watched the movie
# this is the kind of input accepted by BM Neural Network
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users] # takes movies ids only for particular user
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))

    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors (tensors are simply multi-dimensional matrix)
# can be done by tenorflow but Pytorch performed better
training_set = torch.FloatTensor(training_set) # converts nparray into torch tensor
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
class RBM():
    # default function __init__ which runs after every class initialization
    def __init__(self, nv, nh): # nv = no. of visible nodes, nh = no. of hidden nodes
        self.W = torch.randn(nh, nv) # initialises tensor of size nhnv, and according to normal distibution with mean = 0
        self.a = torch.randn(1, nh) # 1 increases the dimension and contains batches of given size
            # this bais of hidden nodes = probability of hidden nodes given visible nodes
        self.b = torch.randn(1, nv) # this bais of visible nodes = probability of visible nodes given hidden nodes

    def sample_h(self, x): # returns samples of hidden nodes of the rbm, p(H|V) = probability of activation function
        # x = values visible nodes in p(H|V)
        wx = torch.mm(x, self.W.t()) # makes products of two tensors, t is transpose more making product consistent
        activation = wx + self.a.expand_as(wx) # expand corrects the dimensionality issue as a as more than 1 dimension
        p_h_given_v = torch.sigmoid(activation) # probability of hidden node being activated given the value of v
        
        return p_h_given_v, torch.bernoulli(p_h_given_v) # samples of hidden nodes given the value of p_h_given_v of those nodes (bernoulli Sampling)

    def sample_v(self, y): # returns samples of visible nodes of the rbm, p(V|H) = probability of activation function
        # y = values hidden nodes in p(V|H)
        wy = torch.mm(y, self.W) # as W is already p(V|H) transpose is not required
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation) # probability of visible node being activated given the value of h
        
        return p_v_given_h, torch.bernoulli(p_v_given_h) # samples of visible nodes given the value of p_v_given_h of those nodes

    # Contrastive Divergence (Converging towards minimum energy with Gibbs Sampling)
    def train(self, v0, vk, ph0, phk): # v0 = input vector = ratings for all movies for a user
        # vk = visible nodes obtained after k samplings(round trips)
        # ph0 = probability that value of hidden nodes = 1 given the value of v0 after 1 iteration
        # phk = probability that value of hidden nodes = 1 given the value of vk after k iterations
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0) # 0 is used for maintaining the 2 dimensions of b
        self.a += torch.sum((ph0 - phk), 0)

# Intializing the RBM class
nv = len(training_set[0]) # no. of movies
nh = 100 # no. of features in all movies, can be any relevant number, model improves on tuning this number
batch_size = 100 # tunable

rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10 # as only binary outcome convergence would be achieved pretty quickly
for epoch in range(1, nb_epoch + 1):
    train_loss = 0 # calculating loss as epochs go on
    s = 0. # floating value
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user: id_user + batch_size]
        v0 = training_set[id_user: id_user + batch_size]
        ph0, _ = rbm.sample_h(v0) # , _ makes python understand that we only need first returned value
        for k in range(10):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0]
        
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0])) # abs distance between actual value and predicted values
        s += 1.

    print('Epoch: ' + str(epoch) + ', Loss: ' + str(train_loss/s)) # train_loss/s normalized loss

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user: id_user + 1] # v = on which predictions will be made
    vt = test_set[id_user: id_user + 1] # vt = target
    if len(vt[vt >= 0]) > 0:
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        s += 1.

print('Test Loss: ' + str(test_loss/s))