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


class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv) # initializes tensors of size nh nv according to normal distribution
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t()) # transpose the weights
        activation = wx + self.a.expand_as(wx) # made sure that bias is applied to every mini batch
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        wy = torch.mm(y, self.W) # no need to transpose as this is the P of v given h and W is already that
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk) # refer to the paper for equation
        self.b += torch.sum((v0 - vk), 0) # use this format to keep the dimensionality
        self.a += torch.sum((ph0 - phk), 0)