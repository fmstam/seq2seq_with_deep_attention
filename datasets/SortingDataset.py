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

from torch.utils.data import Dataset, DataLoader

class SortingDataset(Dataset):

    def __init__(self,
                range_=[0, 100], # min and max numbers to be expected in an instance
                SOS_SYMBOL=-1, # start-of-sequence symbol used for padding the input to the decoder
                lengths=[10, 15, 20],
                gen_indices=[0], # that is, lengths[gen_indeces] instances will be generated
                num_instance=10000):  # for each element in gen_indices

        self.SOS_SYMBOL = SOS_SYMBOL
        self.range = range_
        self.num_instance = num_instance
        self.lengths = lengths
        self.gen_indices = gen_indices
        self.num_instance = num_instance

        self.current_gen_index = gen_indices[0]
        self.instance_counter = 0

    def __len__(self):
        return len(self.gen_indices) * self.num_instance
        
    def __getitem__(self, idx):
        # idx is not used
        # if torch.is_tensor(idx):
        #    idx = idx.item()

        ## keep track of number of generated instance
        self.instance_counter += 1
        ## current length
        length = self.lengths[self.gen_indices[self.current_gen_index]]

        ## generate an instance
        arr = np.random.randint(low=self.range[0], high=self.range[1], size=length)
        sorted_arr_args = np.argsort(arr) # we can obtain the actual easily by arr[sorted_arr_args]
        # shifted target (ugly line but compact)
        shifted_sorted_arr_args = np.concatenate((np.expand_dims(np.array(self.SOS_SYMBOL), axis=1), sorted_arr_args[:-1]), axis=0)
        # check if we need to modify the length of the instance
        if self.instance_counter == self.num_instance:
            self.current_gen_index += 1 # move to the next length
            
        # return a tuple (input, target, shifted_target)
        item =(arr, sorted_arr_args, shifted_sorted_arr_args)        

        return item










    


    

