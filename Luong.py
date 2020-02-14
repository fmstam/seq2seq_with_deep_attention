#!/usr/bin/env python
""" 
    Luong seq2seq global attention approch
    https://arxiv.org/abs/1508.04025
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

#
from seq2seq_with_deep_attention.RDN import EmbeddingLSTM


class LuongGlobalAttention(nn.Module):
    def __init__(self,  
                 num_embeddings,
                 hidden_size,
                 output_size,
                 batch_size,
                 sos_symbol_index,
                 lr = 1e-4,
                 device='cpu'):

        super(LuongGlobalAttention, self).__init__()

        # attributes
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.sos_symbol_index = torch.tensor(sos_symbol_index)
        self.lr = lr
        # device
        if device is not 'cpu':
            if torch.cuda.is_available():
                device = 'cuda:0'
        self.device = device

        # encoder/decoder
        self.encoder = EmbeddingLSTM(self.num_embeddings,
                               batch_size=self.batch_size,
                               hidden_size=self.hidden_size,
                               device=self.device
                               )
        self.decoder = EmbeddingLSTM(self.num_embeddings,
                               hidden_size=self.hidden_size,
                               batch_size=self.batch_size,
                               device=self.device)

        # since we concatenate the h states of the decoder with the context the output is twice the hidden size
        self.cf = nn.Linear(self.hidden_size * 2, self.output_size) 

        #self.loss_function = nn.NLLLoss()
        #self.opitmizer = optim.adam(self.parameters(), lr=self.lr)

        self.to(self.device)
        
    def forward(self, x):
        """ Returns the generated sequence and the accumlated NLL loss 

        keyword argumenents:
        x -- the input sequence
        hidden -- starting hidden state
        """
        ## 1 -  get the encoder output
        encoder_output, hidden = self.encoder(x)

        ## 2 -  get the decoder output
        # loop unitl we recieve all words
        # create decoder input
        decoder_input = self.sos_symbol_index # initial start of sequence symbol
        decoder_input = decoder_input.unsqueeze(-1).unsqueeze(-1) # 3 dimensions
        for i in range(self.output_size):
            decoder_output, _ = self.decoder(decoder_input, hidden, clear_state=False)
            # calcualte attention with dot product and get probs
            attention = F.softmax(torch.bmm(encoder_output, decoder_output))
            # calculate the context
            context = torch.bmm(encoder_output, attention)
            # concatenate contxt with decoder output
            classifer_input = torch.cat(context, decoder_output)
            # do the classfication step
            next_symbol_probs = F.log_softmax(self.fc(classifer_input))
            # get the 
            _, next_symbol = next_symbol_probs.max()
            decoder_input.append(next_symbol)
        




            





                        
        

        
    

