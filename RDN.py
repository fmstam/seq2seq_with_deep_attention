#!/usr/bin/env python
""" 
    Implementation of the RDN components
"""
__author__ = "AL-Tam Faroq"
__copyright__ = "Copyright 2020, UALG"
__credits__ = ["Faroq AL-Tam"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Faroq AL-Tam"
__email__ = "ftam@ualg.pt"
__status__ = "Production"


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 num_embeddings,
                 embedding_dim,
                 device = 'cpu'
                 ):

        super(self, Encoder).__init__()

        # attributes
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_embeddings = input_size # duplicate but more readable
        self.embedding_dim = hidden_size # ...

        # device
        if device is not 'cpu':
            if torch.cuda.is_available():
                device = 'cuda:0'
        self.device = device

        # layers
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(input_size, hidden_size)

        # put the model in the device
        self.to(self.device)


    def forward(self, x, init_hidden_state_and_cell):
        """ The encoder size which takes:

        keyward arguments:
        x -- input
        init_hidden_state_and_cell -- (tuple) the initial hidden state and hidden cell state
        """

        x = torch.tensor(x)
        embd = self.embedding(x) # first step
        encoder_output, init_hidden_state_and_cell = self.lstm(embd, init_hidden_state_and_cell)

        return encoder_output, init_hidden_state_and_cell






    


