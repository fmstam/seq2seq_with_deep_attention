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
                 sequence_length=12):  #

        self.json_file = json_file
        self.get_index = get_index
        self.sequence_length = sequence_length


        # load
        print('Loading dataset...')
        self.dataset = sum(json.loads(open(self.json_file).read()), [])
        print('done!')

        # build a dictionary
        print('Building dictionay ...')
        lst = []
        [lst.extend([w for w in x['input']]) for x in self.dataset]
        self.vocab = set(lst)
        self.word_to_index = {word: i for i, word in enumerate(self.vocab)}
        print('done')
        

    def __len__(self):
        return len(self.dataset)
    
    def sequence_to_index(self, seq):
          # input size, checkup
        if len(seq) > self.sequence_length:
            seq = seq[:self.sequence_length]
        if len(seq) < self.sequence_length:
            l = len(seq)
            st = ''
            seq = seq + st.join([' ' for _ in range(self.sequence_length-l)])
            
        return torch.tensor([self.word_to_index[w] for w in seq])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        item = self.dataset[idx]
        
        if self.get_index: # index in the word_to_index lookup table
            in_seq = self.sequence_to_index(item['input'])
            item = (in_seq, item['output'])
        
        return item










    


    

