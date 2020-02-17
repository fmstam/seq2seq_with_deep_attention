#!/usr/bin/env python
""" 
    Helper classes and functions
"""
__author__ = "AL-Tam Faroq"
__copyright__ = "Copyright 2020, UALG"
__credits__ = ["Faroq AL-Tam"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Faroq AL-Tam"
__email__ = "ftam@ualg.pt"
__status__ = "Production"

import sys
sys.path.append("..")



# local files
from seq2seq_with_deep_attention.helpers import DateDataset, get_sequence_from_indexes
from seq2seq_with_deep_attention.Luong import LuongGlobalAttention

# torch
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)

# constants
INPUT_SIZE = 12
OUTPUT_SIZE = 10
HIDDEN_SIZE = 2
BATCH_SIZE = 1
SOS_SYMBOL = '\t' # start of sequence symbol
EOS_SYMBOL= '\n'
PADDING_SYMOBL = '_'


import matplotlib.pyplot as plt


def main():
 
    ds = DateDataset('out.json', 
                     get_index=True,
                     sequence_length=INPUT_SIZE,
                     SOS_SYMBOL=SOS_SYMBOL,
                     PADDING_SYMBOL=PADDING_SYMOBL)

    dataloader = DataLoader(ds,
                            batch_size=BATCH_SIZE,
                            shuffle=True, 
                            num_workers=0)


    loung = LuongGlobalAttention(input_num_embeddings=len(ds.input_vocab),
                                 output_num_embeddings=len(ds.output_vocab),
                                 hidden_size=HIDDEN_SIZE,
                                 output_size=len(ds.output_vocab),
                                 batch_size=BATCH_SIZE,
                                 sos_symbol_index=ds.input_word_to_index[SOS_SYMBOL],
                                 padding_symbol_index=ds.input_word_to_index[PADDING_SYMOBL],
                                 device='gpu')

    
    # loss function and optimizer
    loss_function = nn.NLLLoss()
    opitmizer = optim.Adam(loung.parameters(), lr=0.001)

    losses = []
    samples = []
    for batch, out_seq in dataloader:
        out_seq = out_seq.to(loung.device)
        # train a Loung seq2seq model
        loung.zero_grad()
        loung.encoder.zero_grad()
        loung.decoder.zero_grad()
        output_seq_probs, output_seq, hidden, attention, context = loung(batch, out_seq)
        loss = 0
        for i in range(OUTPUT_SIZE):
            loss += loss_function(output_seq_probs[:, i, :], out_seq[:, i])
        loss.backward()
        opitmizer.step()
        losses.append(loss.detach().cpu().item())
        #samples.append((get_sequence_from_indexes(ds.input_word_to_index, batch[0,:]), get_sequence_from_indexes(ds.output_word_to_index, output_seq.squeeze(-1))))
        samples.append((out_seq.detach().cpu().numpy(), output_seq.detach().cpu().numpy()))

    # save results
    with open('samples.txt','w') as f:
        f.write(str(samples))   

    # plot loss
    plt.plot(loss)
    plt.plot(losses)
    plt.plot()
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.show()

if __name__ is '__main__':
    main()