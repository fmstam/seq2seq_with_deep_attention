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

class EmbeddingLSTM(nn.Module):
    """ A generic LSTM encoder """
    def __init__(self,
                 num_embeddings,
                 batch_size,
                 device,
                 embedding_layer=nn.Embedding, # nn.Embedding, nn.Linear, nn.Conv1d, ...
                 hidden_size=64, # features
                 ):

        super().__init__()

        # attributes
        self.num_embeddings = num_embeddings
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.device = device
  
        # layers
        self.embedding = embedding_layer(num_embeddings=self.num_embeddings, 
                                      embedding_dim=self.hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        # put the model in the device
        self.to(self.device)

    
    def forward(self, x, hidden_state_and_cell=None, clear_state=True, init_random=False):
        """ The encoder size which takes:

        keyward arguments:
        x -- input, expected to be a tensor being loaded in the same device as the model
        hidden_state_and_cell -- (tuple) the initial hidden state and hidden cell state
        clear_state -- If True then hidden_state_and_cell will be initialized according to init_random
        init_random -- If True then hidden_state_and_cell will be initialized with randoms, zeros otherwise
        """

        # hidden state initialization
        if clear_state:
            if init_random:
                state = torch.randn(1, self.batch_size, self.hidden_size)
            else:
                state = torch.zeros(1, self.batch_size, self.hidden_size)
            state = state.to(self.device)
            hidden_state_and_cell = (state, state)

        # assert hidden_state_and_cell is not None, 'LSTM state should be initialized'
     
        x = x.to(self.device)
        embd = self.embedding(x) # first step
        encoder_output, hidden_state_and_cell = self.lstm(embd, hidden_state_and_cell)

        return encoder_output, hidden_state_and_cell
    




    






    


