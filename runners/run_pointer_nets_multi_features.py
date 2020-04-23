###!/usr/bin/env python
#%% Pointer networks example

""" 
Pointer network example:

This example shows how to use pointer networks to sort numbers.
In this case, we use multiple features as input to the decoder and decoder parts.
Therefore, each instance in the dataset is a tuple ((array, weights), sorted_array).
For more details, see the SortingDataset class.

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
from seq2seq_with_deep_attention.models.MaskedPointerNetwork import MaskedPointerNetwork

# torch
import torch
import torch.nn as nn
import torch.functional as F
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
IN_FEATURES = 2 
HIDDEN_SIZE = 256
BATCH_SIZE = 64
RANGE = [0, 100] # range of generated numbers in a sequence
SOS_SYMBOL = 1.0 # start of sequence symbol 
DATASET_SIZE = 2000
EPOCHS = 50


def plot_attention(attention, input_word, generated_word, size_=(10,10)):
    """ Plot the attention matrix between the input and output sequences.
    """
    print('\nAttention matrix')
    # plot last attention
    plt.matshow(attention)
    plt.xlabel('generated sequence')
    plt.xticks(range(size_[0]),generated_word)
    plt.ylabel('input sequenece')
    plt.yticks(range(size_[1]),input_word)
    plt.show(block=False)


def main():

    # dataset generator
    ds = SortingDataset(use_weights= True, range_=RANGE, SOS_SYMBOL=SOS_SYMBOL, num_instances=DATASET_SIZE)    
    
    
    # loaders
    train_dataloader = DataLoader(ds,
                            batch_size=BATCH_SIZE,
                            num_workers=0)


    # The Masked Pointer Network model
    pointer_network = MaskedPointerNetwork(in_features=IN_FEATURES,
                                 hidden_size=HIDDEN_SIZE,
                                 batch_size=BATCH_SIZE,
                                 sos_symbol=SOS_SYMBOL,
                                 device='gpu')

    
    # loss function and optimizer
    loss_func = nn.MSELoss()
    opitmizer = optim.Adam(pointer_network.parameters(), lr=0.00025)

    ################## Training #############
    #torch.autograd.set_detect_anomaly(True)
    print('Training ...')
    pointer_network.train()
    epochs_loss = []
    for _ in range(EPOCHS):
        losses = []
        for batch,  target_seq in train_dataloader:
            
            # put them in the same device as the model
            batch = batch.float().to(pointer_network.device)
            target_seq = target_seq.to(pointer_network.device)

            # fix dims order to (batch_size, seq_size, features)        
            batch = batch.permute(0,2,1) 

            # handel last batch size problem
            last_batch_size, sequence_length,_ = batch.shape
            pointer_network.update_batch_size(last_batch_size)

            # zero grad        
            pointer_network.zero_grad()
            pointer_network.encoder.zero_grad()
            pointer_network.decoder_cell.zero_grad()
       
            # apply model
            attentions, pointers = pointer_network(batch)

            # loss calculation
            loss = 0
            # can be replaced by a single elegant line, but I do it like this for better readability
            # the one_hot can be moved to the dataset for a better optimization of resources
            for i in range(sequence_length):
                loss += loss_func(attentions[:, i, :].to(pointer_network.device), \
                         nn.functional.one_hot(target_seq[:, i], num_classes=ds.lengths[0]).float()) 
            #backpropagate
            loss.backward()
            opitmizer.step()

            # loss curve
            losses.append(loss.detach().cpu().item())

            # uncomment this line to store all training tuples
            #samples.append((target_seq.detach().cpu().numpy(), pointers.detach().cpu().numpy()))  
        epochs_loss.append(sum(losses) / len(losses))

    # plot loss
    plt.figure()
    plt.title("Training")
    plt.plot(epochs_loss)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.show()

    ################## Testing #############
    pointer_network.eval() # trun off gradient tracking
    test_sequence_length = 12
    test_batches = 1 # one batch for testing
    print('\n\n\nTesting using  a higher length %d'% test_sequence_length)
    
    ds = SortingDataset(use_weights=True, 
                        range_=RANGE,
                        lengths=[test_sequence_length], 
                        SOS_SYMBOL=SOS_SYMBOL, 
                        num_instances=test_batches*last_batch_size)

    test_dataloader = DataLoader(ds,
                            batch_size=last_batch_size,
                            num_workers=0)
    print('\ninput\tweights\ttarget\tpointer')
    for batch, target_sequences in test_dataloader:

        # fix dims order to (batch_size, seq_size, features)        
        batch = batch.permute(0,2,1) 
        batch = batch.float().to(pointer_network.device) # add another dim for features 
        attentions, pointers = pointer_network(batch)

        pointers = pointers.detach().cpu().numpy().astype(int)
        input_sequences = batch.squeeze(2).detach().cpu().numpy()
        i = 0
        for input_seq, target_seq, pointer in zip(input_sequences, target_sequences, pointers):
            print(input_seq[:,0], input_seq[:,1], input_seq[target_seq,0], input_seq[pointer,0])
            plot_attention(attentions[i].t().detach().cpu().numpy(), input_seq[:,0].astype(int), input_seq[pointer, 0].astype(int), size_=(test_sequence_length, test_sequence_length))
            i += 1


if __name__ is '__main__':
    main()

# %%


