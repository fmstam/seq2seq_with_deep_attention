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

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class LuongGlobalAttention(nn.Module):
    def __init__(self,  
                 input_num_embeddings,
                 output_num_embeddings,
                 hidden_size,
                 output_size,
                 batch_size,
                 sos_symbol_index,
                 padding_symbol_index,
                 device='cpu'):

        super(LuongGlobalAttention, self).__init__()

        # attributes
        self.input_num_embeddings = input_num_embeddings
        self.output_num_embeddings = output_num_embeddings
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.sos_symbol_index = torch.tensor(sos_symbol_index)
        self.padding_symbol_index = torch.tensor(padding_symbol_index)
        # device
        if device is not 'cpu':
            if torch.cuda.is_available():
                device = 'cuda:0'
        self.device = device

        # encoder/decoder
        self.encoder = EmbeddingLSTM(self.input_num_embeddings,
                               batch_size=self.batch_size,
                               hidden_size=self.hidden_size,
                               device=self.device
                               )
        self.decoder = EmbeddingLSTM(self.output_num_embeddings,
                               hidden_size=self.hidden_size,
                               batch_size=self.batch_size,
                               device=self.device)

        # since we concatenate the h states of the decoder with the context the output
        #  is twice the hidden size, and the output is the same size as the length of the 
        #  output vocabs.
        self.fc = nn.Linear(self.hidden_size * 2, self.output_size) 

        self.to(self.device)


    def forward(self, input_seq, target_seq):
        """ Returns the generated sequence and the accumlated NLL loss 

        keyword argumenents:
        input_seq -- the input sequence
        target_seq -- the output sequence
        hidden -- starting hidden state
        """

        # get the encoder output
        encoder_output, hidden = self.encoder(input_seq)
        # get the decoder output
        decoder_output, hidden = self.decoder(target_seq, hidden, clear_state=False)
        # fix the dimension of the encode_output
        encoder_output_ = encoder_output.permute(0, 2 , 1)
        # calcualte attention with dot product and get probs
        attention = F.softmax(torch.bmm(decoder_output, encoder_output_), dim=2)
        # calculate the context
        context = torch.bmm(attention, encoder_output)
        # concatenate contxt with decoder output
        classifer_input = torch.cat((context, decoder_output), dim=2)
        # do the classfication step
        #fc_out = self.fc(classifer_input)
        #fc_out = self.fc(torch.tanh(classifer_input))
        td = TimeDistributed(self.fc, batch_first=True)
        fc_out = td(torch.tanh(classifer_input))
        output_seq_probs = F.log_softmax(fc_out, dim=2)
        #output_seq_probs = F.softmax(fc_out, dim=2)
        # get the 
        _, output_seq = output_seq_probs.max(dim=2)

        return output_seq_probs, output_seq, hidden, attention, context
        #decoder_input.append(next_symbol)
        




            





                        
        

        
    

