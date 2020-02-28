#!/usr/bin/env python
""" 
    Pointer Networks
    https://arxiv.org/abs/1506.03134
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

import numpy as np

from seq2seq_with_deep_attention.RDN import EmbeddingLSTM

class PointerNetwork(nn.Module):
    def __init__(self,  
                in_features,
                hidden_size,
                batch_size,
                device='cpu'):

        super(PointerNetwork, self).__init__()

        # attributes
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        # device
        if device is not 'cpu':
            if torch.cuda.is_available():
                device = 'cuda:0'
        self.device = device

        # encoder/decoder
        self.encoder = EmbeddingLSTM(embedding_layer=nn.Linear(in_features=in_features, out_features=hidden_size),
                                batch_size=self.batch_size,
                                hidden_size=self.hidden_size,
                                device=self.device
                                )
        # in figure 1b, we need to perfrom the decorder step by step
        # this is a curcial piece of pointer network and the main difference with seq2seq models
        # see the forward function for more details
        self.decoder_cell = EmbeddingLSTM(embedding_layer=nn.Linear(in_features=in_features, out_features=hidden_size),
                                batch_size=self.batch_size,
                                hidden_size=self.hidden_size,
                                device=self.device,
                                lstm_cell=True # LSTMCell
                                )

        # attention calculation paramaters see first lines in equation 3 in 
        # u^i_j = v^\top tanh(W_1 e_j + W_2 d_i), \forall j \in (1, \cdots, n)
        # where e_j and d_i are the encoder and decoder hidden states, respectively.
        # W_1, and W_2 are learnable weights(square matrices), we represent them here by nn.Linear layers
        # v is also a vector of learnable weights, we represent it here by nn.Paramater

        self.W_1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.W_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.v = nn.Linear(in_features=hidden_size, out_features=1)

        self.to(self.device)

    def forward(self, input_seq):
        """
        Calculate the attention
        
        keyword argumenents:
        input_seq -- the input sequence (batch_size, sequence_size, hidden_size)
        shifted_target_seq -- the output sequence shifted one symbol (batch_size, sequence_size, hidden_size)
        """
        _, input_seq_length,_ = input_seq.shape
        ## get the encoder output
        encoder_output, hidden = self.encoder(input_seq)

        ## get the decoder output
        # 1- we start by inserting the random or zeros to the encoder_cell
        # 2- the pointer network will produce a pointer from the softmax activation
        # 3- We use that pointer to select the embedded features from the input sequence
        # 4- we feed these selected features to the encoder_cell and get a new pointer
        # 5- we iterate the above steps until we collect pointers with the same size as the input
        
        # we will use them to calculate the loss function
        pointers = []
        pointers_scores =[]

        # the initial state of the decoder_cell is the last state of the encoder
        decoder_cell_hidden = (hidden[0][-1, :, :], hidden[1][-1, :, :])  # each of size(num_layers=1, batch_size, hidden_size)
       
        # initialize the first input to the decoder_cell, zeros or random
        #decoder_cell_input = torch.rand((self.batch_size, 1)) # one is for the feature not the step
        decoder_cell_input = torch.zeros((self.batch_size, 1)) # one is for the feature not the step

        for _ in range(input_seq_length):
            # 1 - calculate decoder hidden and cell states 
            decoder_cell_output, decoder_cell_hidden = self.decoder_cell(decoder_cell_input, decoder_cell_hidden, clear_state=False)
            # 2 - used decoder_cell_output and encoder_output to calculate the attention:
            # u^i_j = v^\top tanh(W_1 e_j + W_2 d_i), \forall j \in (1, \cdots, n)
            # size of u is (batch_size, sequence_length, 1)
            # so we remove that last dimenstion 
            u = self.v(torch.tanh(self.W_1(encoder_output) + self.W_2(decoder_cell_output).unsqueeze(1))).squeeze(2)
            
            # # a^i_j = softmax(u^i_j)
            attention = F.log_softmax(u) # we use the log for two reasons:
            #                              # 1- avoid doing hot_one encoding
            #                              # 2- mathematical stability
        

        # return attention, hidden, decoder_output


