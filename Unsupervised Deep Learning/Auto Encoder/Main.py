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
    sep = '::',
    header = None,
    engine = 'python',
    encoding = 'latin-1')

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
    delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv(r'D:\Learning\Udemy ML\MLPracticeProjects\Unsupervised Deep Learning\ml-100k\u1.test',
    delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Converting the data into an array for training_set and test_set
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))

    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors (tensors are simply multi-dimensional matrix)
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module): # SAE for Stacked Auto Encoder, we are inherting from nn.Module class
    def __init__(self, ): # after , is space as it will take paramters from inherited class
        super(SAE, self).__init__() # this takes in all classes and functions of inherited class
        self.fc1 = nn.Linear(nb_movies, 20) # first full connection between input and hidden layer
            # 1st parameter is number of input features
            # 2nd is changable number of hidden nodes in a layer, 20 is based on trial and error
        self.fc2 = nn.Linear(20, 10) # second layer
            # 1st is number of neurons of 1st layers
            # 2nd is changable number of hidden nodes in a layer, 10 is based on trial and error
        self.fc3 = nn.Linear(10, 20) # here 2nd para is 20 as we start decoding
        self.fc4 = nn.Linear(20, nb_movies) # output layer
        self.activation = nn.Sigmoid() # activation function to be used in neurons, it is tunable, can use Rectifier as well

    def forward(self, x): # Applying Encoding-Decoding, x: input vector
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x)) # from here starts decoding
        x = self.fc4(x) # no activation function is used, predicted ratings

        return x

sae = SAE()
criterion = nn.MSELoss() # Meas Squared Error Loss Function
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # not adam this time, as this gives better results
    # lr is tunable
    # decay is used to reduced the learning rate after every few epochs, its tunable and improves the model

# Training the SAE (steps below can be used for bigger datasets as well as this is well optimized)
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0. # intially is used to compute only for users to rated atleast 1 movie
        # made float so that it can be further used in mse
    for id_user in range(nb_users):
        input_vector = Variable(training_set[id_user]).unsqueeze(0) # increases dimenion as nn layers do not accept 1 dimension data
        target = input_vector.clone() # copying input as target would be modified going forward
        if torch.sum(target.data > 0) > 0: # checking user if they atleast rated 1 movie
            output = sae.forward(input_vector)
            target.require_grad = False # so that gradient is not computed with respect to target and done only for input
            output[target == 0] = 0 # for movies that were not rated by the user and these will not be included in computations
                # will also help in optimzation of performance
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # taking movies having non-zero rating
                # 1e-10 is added so that denominator is never 0 and is very small so won't effect the calculations
                # mean_corrector is calculated so that mean is calculated only for non-zero ratings
                # this would useful while calculation mean of errors
            loss.backward()
            train_loss += np.sqrt(loss.data * mean_corrector) # sqrt gives better error representation
            s += 1.
            optimizer.step() # applies the optimizer to update the weights
                # backward decides the direction to which the weights would be updated, increase or decrease
                # while the optimizer decides the intensity of the updates, i.e. the amount with which the weights would be updated

    print('Epoch: ' + str(epoch) + ', Loss: ' + str(train_loss / s)) # loss close to 1 means making wrong prediction of 1 rating

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input_vector = Variable(training_set[id_user]).unsqueeze(0) # used training_set as predicting on movies user hasn't rated yet
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0: # checking user if they atleast rated 1 movie
        output = sae.forward(input_vector)
        target.require_grad = False
        output[target == 0] = 0 # there might still be some movies that were not rated 
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data * mean_corrector)
        s += 1.

print('Test Loss: ' + str(test_loss / s))