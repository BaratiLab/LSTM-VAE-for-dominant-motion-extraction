import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder 

    Param:
    number_of_features: number of input features
    hidden_size: hidden size
    hidden_layer_depth: number of layers
    latent_length: latent vector length
    
    """
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.dropout=dropout
        
        self.model = nn.LSTM(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = self.dropout)
        

    def forward(self, x):
        """Forward propagation of encoder. Given input, outputs the last hidden state of encoder
        Param:
        x: input shape (sequence_length, batch_size, number_of_features)
        
        Return: 
        last hidden state, shape (batch_size, hidden_size)
        """

        _, (h_end, c_end) = self.model(x)

        h_end = h_end[-1, :, :]
        return h_end
