# boltzmann machines (specifically Restricted Boltzmann Machines)

# import the required libraries
import numpy as np
import pandas as pd

# import all pytorch libraries
import torch

# torch.nn is for neural networks
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# import dataset now
# since the title of the movies has commas, we cannot have the separator as commas
# we use :: instead in .dat files
# then we need to specify the headers
# need to specify encoding as well as movies contain special characers cannot be handled by UTF8
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# create variable for users
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# import the ratings
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# prepare the training set and test set
# generally we have multiple train-test splits so that we can perform k-test cross validation
# if separator is a tab then it's better to use delimiter instead of sep
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')

# training_set nust convert into array
# we have to create array of integers
training_set = np.array(training_set, dtype = 'int')

# now we import the test sets
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# we need to get the number of users and movies
# as we need to convert into a matrix where row = users, columns = movies, cells = ratings
# if a user didn't rate a movie then we'll put a 0 in the matrix
# matrix for test and training set. same number of lines and columns
# each cell will have a rating the movie got for every user
# code needs to applicable with other sets
# need to get max user id and max no of movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0]))) # take first column
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1]))) # take second column

# then we need to convert into a matrix/array with users
# restricted boltzmann machines require specific type of data types that contain
# these neural networks. Observations in lines, features in columns, must be in this format
# now users in lines and movies in columns
# create function for it
def convert(data):
    # create list of lists
    # first list is first user, second lsit sceond user etc
    new_data = []
    # for loop the new data
    # remember upper bound is excluded
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users] # second bracket is the condition for the data in the first bracket
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings # have to reduce by 1 as array starts at 0
        new_data.append(list(ratings))
    
    return new_data        

training_set = convert(training_set)
test_set = convert(test_set)

# now we need to convert this data into torch tensors, the input into pytorch
# we will start creating the architecture soon (pyTorch tensors)
# tensors are arrays that contain elements of a single data type
# tensors are multi-dimensional matrix, must make a pyTorch array not numpy array
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# then we need to convert all ratings into binary ratings 1 (liked) or 0 (not liked)
# specific to boltzmann machines is we have to convert into binary ratings
# because we want to predict binary ratings as well, RBM will take the input vector
# it will predict that was not originally rated by the user. Predicted ratings must have
# same format
# convert all 0 into -1 as 0 is taken by another definition

# take all values of traiing set such that values of traning set is 0
# not liked is 1 star or 2 stars
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0 # the operator or doesn't work in pytorch tensors
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

# convert test set as well
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0 # the operator 'or' doesn't work in pytorch tensors
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# now input is ready to put inside RBM
# build the architecture of the neural network
# make a probabilistic graphical model as RBM is exactly this type of model
# define class to build RBM
# no of hidden nodes, weights, bias for probability of visible node given hidde node
# make 3 functions, create RBM object, sample the probabiltiy of the hidden nodes
# given the visible nodes and vice versa
class RBM():
    # start with init function to instantiate all the variables
    # specify the variables, nv is no of visible nodes, nh is no of hidden nodes
    # __init__ commences after an object of this type is initialized
    def __init__(self, nv, nh):
        # initialize all the parameters which we will optimize in the RBM, i.e. weights and bias
        # specify that these variables belong to the class
        # all weights are initialized in a torch tensor, weights are all the parameters
        # of the probabilities of the visible nodes wrt to the hidden nodes
        self.W = torch.randn(nh, nv) # initializes tensors of size nh nv according to normal distribution
        # initialize the bias probab of hidden nodes given the vis nodes and vice versa
        # must put all the variables into 2D vectors as the torch expects it
        # nh creates the bias
        self.a = torch.randn(1, nh)
        # bias for visible nodes, the first dimension corresponds to the batch
        self.b = torch.randn(1, nv)
    
    # we need to then sample h function (sigmoid function) during the training we will
    # approximate the log likelihood gradients. We can sample the activations of the hdiden nodes then
    # for each of the 100 nodes we'll activate them according to some probability (ph given v)
    # x refers to the visible neurons v in the probabilities ph given v
    def sample_h(self, x):
        # compute probability h given v, input vector of observations with all the ratings
        # no other than sigmoid function
        # mm gives the product of 2 matrices
        wx = torch.mm(x, self.W.t()) # transpose the weights
        # since the vector includes batches, we want to make sure that everything is applied
        # to each line to every mini batch
        activation = wx + self.a.expand_as(wx) # made sure that bias is applied to every mini batch
        # since activation is a probability
        p_h_given_v = torch.sigmoid(activation)
        # then we return all the sample of the hidden neurons
        # make a bernoulli rbm
        # for high p_h_given_v for a node we need to activate the neuron
        # take rondom number between 0 and 1 if >0.7 then activate (0.7 is arbitrary, depends on the features calculated by RBM)
        # if < 0.7 then never activate
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    # similar to above, need to return bernoulli sampling
    # we have 1682 visible nodes, need to return 1682 probabilities as well
    # each probability is the probability the visible nodes correspond to 1
    def sample_v(self, y):
        wy = torch.mm(y, self.W) # no need to transpose as this is the P of v given h and W is already that
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    # last function is the contrastive divergence to calculate the gradient
    # termed as RBM Log-Likelihood Gradient
    # Since we need to minimize the energy, we need to optimize the weights to
    # minimise the energy, goal is to maximize Log-Likelihood
    # we need to try to updating the weights bit by bit in the direction of the smallest energy (contrastive divergence)
    # Gibbs chain is used
    # v0 is input vectors
    # vk is the visible nodes obtained after iterations of contrastive divergence
    # ph0 is vector of probabilities that at first iteration hidden nodes = 1 at v0
    # phk is probabilities of subsequent k iteration
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk) # refer to the paper for equation
        self.b += torch.sum((v0 - vk), 0) # use this format to keep the dimensionality
        self.a += torch.sum((ph0 - phk), 0)
        
# call object RBM()
# nv is the number of movies (ratings to be exact)
# training_set[0] is first column and put len to find the length
nv = len(training_set[0])

# number of hidden nodes is a parameter we choose
# choose a relevant number, since we have 1682 moveis
# nh is features that are to be detected in these 1682 movies
# e.g. genres, actors, directors, got an oscar, old/new? got a lot
# so no of hidden nodes is the number of features, let's say 100 features then we can try other models
nh = 100

# another parameter not highlighted is the batch size
# when we train our algorithm we won't update after one observation but after multiple observations (in batches)
# start with some batch size, this is totally tunable
batch_size = 100

rbm = RBM(nv, nh)

# we need to now train the RBM now
# choose number of epochs
# since small sample size 10 is enough
nb_epoch = 10

# create for loop to update the weights every epoch
for epoch in range(1, nb_epoch + 1):
    # first we have to measure the error between the real and the predictions, calculate loss
    # RMSE or Absolute Distance?
    # for this RBM we'll use the Absolute Distance
    train_loss = 0
    # create a counter to normalize the loss, must be a float
    s = 0.
    # since our function only does it user by user, we need to do it for all users
    # since this no of trainings depends on the batch_size, we need to take it into consideration as well
    # third param is the step
    for id_user in range(0, nb_users - batch_size, batch_size):
        # input is the ratings of all the movies, target is the same as the input (but will go into the Gibbs chain)
        # input will change but target will be the same value
        vk = training_set[id_user:id_user + batch_size] # the input batch of observations initially
        # target is the original ratings so that we can compare
        # then input will be updated
        v0 = training_set[id_user:id_user + batch_size]
        # then we need to get the initial probabilites
        # ph0 is prob of hiden node = 1 given the initial ratings
        # ,_ returns the first value of the function
        ph0,_ = rbm.sample_h(v0)
        # create another for loop to simulate k-steps of contrastive divergence
        # we'll do 10 k-steps
        for k in range(10):
            # we start with input batch 
            # first step of Gibbs sampling, we're going to sample the first hidden node
            # call sample_h on the visible node, to obtain the first sampling of the first hidden node
            _,hk = rbm.sample_h(vk) # update input nodes only not the target
            # update vk now, after sampling by Gibbs
            _,vk = rbm.sample_v(hk)
            # but we don't want to learn on those with -1 ratings
            # we need to freeze the visible nodes containing -1 ratings
            vk[v0 < 0] = v0[v0 < 0] # makes the -1 the same every iteration
        # now we compute phk, apply sample h on the last sample of the visible node (end of previous loop)
        phk,_ = rbm.sample_h(vk)
        # now we're ready to train our model
        rbm.train(v0, vk, ph0, phk) # weights and bias are going to be updated towards maximum likelihood
        # measure loss now, comparing vk and v0
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        # train_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2)) # here we use RMSE
        # update the counter now, to normalize the train_loss
        s += 1.
    print('epoch:', str(epoch), 'loss:', str(train_loss/s))


# use MCMC to test and validate the RBM
# we don't need a batch size, only specific to training set
# loop over all the users
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user + 1] # keep training set, because this is the input that will be used to activate the hidden neurons to get the outputs
    vt = test_set[id_user:id_user + 1] # target of real test one (original ratings)
    # activate the neurons of the RBM to predict ratings of the movies that were not rated yet
    # the inputs we get from training_set NOT the test_set
    # random walk in MCMC
    # compute probability hidden node equal 1 given the values of the visible nodes of the input vector
    # to get our prediction of the test set ratings, do we need to apply k-step divergence?
    # do we even need to make k steps or one step only? We need to make one step because
    # the principle of a blind walk is testing only with less steps as we trained with multiple steps
    # this is the priniple of MCMC (probabilities are not the same)
    # we need to filter the -1s now, need to get from vt because we're getting from the target
    if len(vt[vt >= 0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_h(h)
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0])) # still use absolute distances
        # test_loss += np.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2)) # here we use RMSE
        s += 1.
print('test loss', str(test_loss/s))

# this was quite easy because we're only using binary 0 or 1 easier and less complex than continuous values

'''
u = np.random.choice([0,1], 100000)
v = np.random.choice([0,1], 100000)
u[:50000] = v[:50000]
sum(u==v)/float(len(u)) # -> you get 0.75
np.mean(np.abs(u-v))

use this to validate the percentage of success
'''
