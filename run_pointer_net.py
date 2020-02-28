###!/usr/bin/env python
#%%
""" 
Example for date conversion
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
from seq2seq_with_deep_attention.datasets.SortingDataset import SortingDataset
from seq2seq_with_deep_attention.models.PointerNetwork import PointerNetwork

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler # to do train-validate spilit

# plot
import matplotlib.pyplot as plt

# utilis
import random
import math



random_seed = torch.manual_seed(45)

# constants
IN_FEATURES = 1 # depends on the demnationality of the input
HIDDEN_SIZE = 64
BATCH_SIZE = 16
RANGE = [0, 100]
SOS_SYMBOL = -1 # start of sequence symbol 

VALIDATION_RATIO = .2


def main():

    ds = SortingDataset(range_=RANGE, SOS_SYMBOL=SOS_SYMBOL)

    # train-validate spilit
    ds_len = len(ds)
    indexes = list(range(ds_len))
    random.shuffle(indexes) # shuffle them
    spilit_spot = int(math.floor(VALIDATION_RATIO * ds_len))
    
    train_indexes = indexes[spilit_spot:]
    validation_indexes = indexes[:spilit_spot]

    # samples 
    train_sampler = SubsetRandomSampler(train_indexes)
    validation_sampler = SubsetRandomSampler(validation_indexes)

    # loaders
    train_dataloader = DataLoader(ds,
                            sampler=train_sampler,
                            batch_size=BATCH_SIZE,
                            num_workers=0)

    validation_dataloader = DataLoader(ds,
                            sampler=validation_sampler,
                            batch_size=BATCH_SIZE,
                            num_workers=0)
    


    # The Pointer Network model
    pointer_network = PointerNetwork(in_features=IN_FEATURES,
                                 hidden_size=HIDDEN_SIZE,
                                 batch_size=BATCH_SIZE,
                                 device='gpu')

    
    # loss function and optimizer
    loss_function = nn.NLLLoss()
    opitmizer = optim.Adam(pointer_network.parameters(), lr=0.001)

    ################## Training #############
    print('Training ...')
    losses = []
    samples = []
    train_for = 5000 # if we wish to train faster for limited number of batches
    for batch, target_seq, target_seq_shifted in train_dataloader:
        if train_for == 0:
            break
        train_for -= 1
        # put them in the same device as the model's
        target_seq_shifted = target_seq_shifted.to(pointer_network.device)
        target_seq = target_seq.to(pointer_network.device)
        # train a pointer_network seq2seq model
        pointer_network.zero_grad()
        pointer_network.encoder.zero_grad()
        pointer_network.decoder_cell.zero_grad()
        batch = batch.unsqueeze(2).float() # input feature domain
        pointer_network(batch)
        #attention, hidden, decoder_output = pointer_network(batch, target_seq_shifted)

        # loss calculation
        loss = 0
        # # can be replaced by a single elegant line, but I do it like this for better readability
        # for i in range(OUTPUT_SIZE):
        #     loss += loss_function(output_seq_probs[:, i, :], target_seq[:, i])
        # back propagate
        loss.backward()
        opitmizer.step()
        # loss curve
        losses.append(loss.detach().cpu().item())
        # uncomment this line to check how store all training tuples
        #samples.append((target_seq.detach().cpu().numpy(), output_seq.detach().cpu().numpy()))  

    # plot loss
    plt.figure()
    plt.title("Training")
    plt.plot(losses)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.show(block=False)

if __name__ is '__main__':
    main()

# %%
