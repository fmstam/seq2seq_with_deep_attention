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

import json # to read the file

# torch
import torch

# tools
import numpy as np

from torch.utils.data import Dataset, DataLoader

class DateDataset(Dataset):
    """ This class read a dataset from a file and create: list of tuples for (input, target) and a dictionay for each word
    """
    def __init__(self, 
                 json_file, 
                 get_index=False,
                 sequence_length=12,
                 SOS_SYMBOL='_ST_',
                 PADDING_SYMBOL='_PA_'):  #

        self.json_file = json_file
        self.get_index = get_index
        self.sequence_length = sequence_length
        self.SOS_SYMBOL = SOS_SYMBOL
        self.PADDING_SYMBOL = PADDING_SYMBOL


        # load
        print('Loading dataset...')
        self.dataset = sum(json.loads(open(self.json_file).read()), [])
        print('done!')

        # build a dictionary
        print('Building dictionay for input and output ...')
        lst = []
        [lst.extend([w for w in x['input']]) for x in self.dataset]
        self.input_vocab = set(lst)
        self.input_vocab.add(self.SOS_SYMBOL) # sequence start special word
        self.input_vocab.add(self.PADDING_SYMBOL) # padding symbol
        self.input_word_to_index = {word: i for i, word in enumerate(self.input_vocab)}

        lst = []
        [lst.extend([w for w in x['output']]) for x in self.dataset]
        self.output_vocab = set(lst)
        self.output_vocab.add(self.SOS_SYMBOL) # sequence start special word
        self.output_word_to_index = {word: i for i, word in enumerate(self.output_vocab)}
        print('done')
        

    def __len__(self):
        return len(self.dataset)
    
    def input_sequence_to_index(self, seq):
          # input size, checkup
        if len(seq) > self.sequence_length:
            seq = seq[:self.sequence_length]
        if len(seq) < self.sequence_length:
            l = len(seq)
            st = ''
            seq = seq + st.join([self.PADDING_SYMBOL for _ in range(self.sequence_length-l)])
            
        return torch.tensor([self.input_word_to_index[w] for w in seq])

    def output_sequence_to_index(self, seq):
        # shift the output sequence by one time step 
        seq = seq[:-1]
        # append an SOS symbol
        seq = self.SOS_SYMBOL + seq
        # get index
        return torch.tensor([self.output_word_to_index[w] for w in seq])


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        # actual entry in the dataset
        item = self.dataset[idx]
        
        # indeces encoding
        if self.get_index: # index in the word_to_index lookup table
            in_seq = self.input_sequence_to_index(item['input'])
            out_seq = self.output_sequence_to_index(item['output'])
            item = (in_seq, out_seq)
        
        return item


def get_sequence_from_indexes(vocab, indexes):
    seq = []
    for x in indexes:
        for k, v in vocab.items():
            if v == x:
                seq.append(k)
    return seq









    


    

