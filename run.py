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
from seq2seq_with_deep_attention.helpers import DateDataset
from seq2seq_with_deep_attention.Luong import LuongGlobalAttention

# torch
import torch
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)

# constants
INPUT_SIZE = 12
OUTPUT_SIZE = 10
HIDDEN_SIZE = 64
BATCH_SIZE = 1
SOS_SYMBOL='_ST_' # start of sequence symbol





def main():
 
    ds = DateDataset('out.json', get_index=True, sequence_length=INPUT_SIZE)
    dataloader = DataLoader(ds,
                            batch_size=BATCH_SIZE,
                            shuffle=True, 
                            num_workers=0)

    # encoder = Encoder(num_embeddings=len(ds.vocab), 
    #                   batch_size = BATCH_SIZE,
    #                   device='gpu')

    loung = LuongGlobalAttention(num_embeddings=len(ds.vocab),
                                 hidden_size=HIDDEN_SIZE,
                                 output_size=OUTPUT_SIZE,
                                 batch_size=BATCH_SIZE,
                                 sos_symbol_index=ds.word_to_index[SOS_SYMBOL],
                                 lr = 1e-4,
                                 device='gpu')

    for batch, out_seq in dataloader:
        # pass to the encoder
        output = loung(batch)
        print(output)


if __name__ is '__main__':
    main()