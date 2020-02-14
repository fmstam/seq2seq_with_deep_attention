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
                 num_embeddings,
                 batch_size,
                 hidden_size=64, # features
                 device = 'cpu'
                 ):

        super().__init__()

        # attributes
        self.num_embeddings = num_embeddings
        self.batch_size = batch_size
        self.hidden_size = hidden_size
  

        # device
        if device is not 'cpu':
            if torch.cuda.is_available():
                device = 'cuda:0'
        self.device = device

        # layers
        self.embedding = nn.Embedding(num_embeddings=self.num_embeddings, 
                                      embedding_dim=self.hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        # put the model in the device
        self.to(self.device)

    


    def forward(self, x, init_hidden_state_and_cell=None, clear_state=True, init_random=False):
        """ The encoder size which takes:

        keyward arguments:
        x -- input, expected to be a tensor being loaded in the same device as the model
        init_hidden_state_and_cell -- (tuple) the initial hidden state and hidden cell state
        clear_state -- If True then init_hidden_state_and_cell will be initialized according to init_random
        init_random -- If True then init_hidden_state_and_cell will be initialized with randoms, zeros otherwise
        """

        if clear_state:
            if init_random:
                state = torch.randn(1, self.batch_size, self.hidden_size)
            else:
                state = torch.zeros(1, self.batch_size, self.hidden_size)
            state = state.to(self.device)
            init_hidden_state_and_cell = (state, state)

        # assert init_hidden_state_and_cell is not None, 'LSTM state should be initialized'
     
        x = torch.tensor(x).cuda()
        embd = self.embedding(x) # first step
        encoder_output, init_hidden_state_and_cell = self.lstm(embd, init_hidden_state_and_cell)

        return encoder_output, init_hidden_state_and_cell






    


