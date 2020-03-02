#!/usr/bin/env python
""" 
    Sorting dataset generator
"""
__author__ = "AL-Tam Faroq"
__copyright__ = "Copyright 2020, UALG"
__credits__ = ["Faroq AL-Tam"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Faroq AL-Tam"
__email__ = "ftam@ualg.pt"
__status__ = "Production"


# torch
import torch

# tools
import numpy as np
import math

from torch.utils.data import Dataset, DataLoader

class SortingDataset(Dataset):

    def __init__(self,
                range_=[0, 100], # min and max numbers to be expected in an instance
                SOS_SYMBOL=-1, # start-of-sequence symbol used for padding the input to the decoder
                lengths=[10, 15, 20],
                gen_indices=[0], # that is, lengths[gen_indeces] instances will be generated
                num_instances=10000):  # for each element in gen_indices

        self.SOS_SYMBOL = SOS_SYMBOL
        self.range = range_
        self.lengths = lengths
        self.gen_indices = gen_indices
        self.num_instances = num_instances

        

        # generating a dataset
        print('Generating a dataset ...')
        l = len(self)
        self.ds = []
        for i in range(l):
            j = math.floor(i / self.num_instances)
            arr = np.random.randint(low=self.range[0], high=self.range[1], size=self.lengths[j])
            sorted_arr_args = np.argsort(arr)
            self.ds.append((arr, sorted_arr_args))


    def __len__(self):
        return len(self.gen_indices) * self.num_instances
        
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.item()

        return self.ds[idx]










    


    

