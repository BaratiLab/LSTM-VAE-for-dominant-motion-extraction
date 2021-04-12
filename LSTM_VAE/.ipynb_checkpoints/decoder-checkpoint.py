import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import torch.nn.functional as F


class Decoder(nn.Module):
   
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size,cuda):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.cuda = cuda

        

        self.model = nn.LSTM(1, self.hidden_size, self.hidden_layer_depth)
        
        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)
        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

        self.decoder_inputs = torch.zeros(self.sequence_length, self.batch_size, 1, requires_grad=True).type(self.cuda)
        self.c_0 = torch.zeros(self.hidden_layer_depth, self.batch_size, self.hidden_size, requires_grad=True).type(self.cuda)
        
    def forward(self, latent):
                
        h_state = self.latent_to_hidden(latent)

        
        h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
        decoder_output, _ = self.model(self.decoder_inputs, (h_0, self.c_0))
        
        out = self.hidden_to_output(decoder_output)
        return out