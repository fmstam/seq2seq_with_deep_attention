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

from seq2seq_with_deep_attention.RDN import EmbeddingLSTM

class PointerNetwork(nn.Module):
    def __init__(self,  
                in_features,
                hidden_size,
                batch_size,
                sos_symbol_index,
                device='cpu'):

        super(PointerNetwork, self).__init__()

        # attributes
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sos_symbol_index = torch.tensor(sos_symbol_index)
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
        self.decoder = EmbeddingLSTM(embedding_layer=nn.Linear(in_features=in_features, out_features=hidden_size),
                                hidden_size=self.hidden_size,
                                batch_size=self.batch_size,
                                device=self.device)

        # attention calculation paramaters see first lines in equation 3 in 
        # u^i_j = v^\top tanh(W_1 e_j + W_2 d_i), \forall j \in (1, \cdots, n)
        # where e_j and d_i are the encoder and decoder hidden states, respectively.
        # W_1, and W_2 are learnable weights(square matrices), we represent them here by nn.Linear layers
        # v is also a vector of learnable weights, we represent it here by nn.Paramater

        self.W_1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.W_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.v = nn.Parameter(torch.Tensor(1, hidden_size))

    def forward(self, input_seq, shifted_input_seq):
        """
        Calculate the attention

        keyword argumenents:
        input_seq -- the input sequence (batch_size, sequence_size, hidden_size)
        shifted_target_seq -- the output sequence shifted one symbol (batch_size, sequence_size, hidden_size)
        """

        ## get the encoder output
        encoder_output, hidden = self.encoder(input_seq)
        ## get the decoder output
        decoder_output, hidden = self.decoder(shifted_input_seq, hidden, clear_state=False)
        # fix the dimension of the encode_output to apply the dot product
        encoder_output_ = encoder_output.permute(0, 2 , 1)
        ## Attention
        # 1 - global scoring: (dot product) calcualte attention with dot product and get probs 
        # not used in the original paper, but is used in attention mechanisim in Loung model
        attention = F.softmax(torch.bmm(decoder_output, encoder_output_), dim=2)

        # 2 - Pointer net mechanisim:
        # u^i_j = v^\top tanh(W_1 e_j + W_2 d_i), \forall j \in (1, \cdots, n)
        # a^i_j = softmax(u^i_j)
        u = torch.bmm(torch.tanh(self.W_1(encoder_output) + self.W_2(decoder_output)))
        attention = F.log_softmax(u) # we use the log for two reasons:
                                     # 1- avoid doing hot_one encoding
                                     # 2- mathematical stability
        

        return attention, hidden, decoder_output


