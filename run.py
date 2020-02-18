###!/usr/bin/env python
#%%
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
from torch.utils.data.sampler import SubsetRandomSampler # to do train-validate spilit

# plot
import matplotlib.pyplot as plt

# utilis
import random
import math



random_seed = torch.manual_seed(45)

# constants
INPUT_SIZE = 12
OUTPUT_SIZE = 10
HIDDEN_SIZE = 64
BATCH_SIZE = 32
SOS_SYMBOL = '\t' # start of sequence symbol
EOS_SYMBOL= '\n'
PADDING_SYMOBL = '_'
VALIDATION_RATIO = .25



def main():
 
    ds = DateDataset('/home/faroq/code/seq2seq/out.json', 
                     get_index=True,
                     sequence_length=INPUT_SIZE,
                     SOS_SYMBOL=SOS_SYMBOL,
                     PADDING_SYMBOL=PADDING_SYMOBL)
    
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
    


    # Loung Model
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


    ################## Training #############
    print('Training ...')
    losses = []
    samples = []
    train_for = 1000 # if we wish to train for limited number of batches
    for batch, target_seq in train_dataloader:
        if train_for == 0:
            break
        train_for -= 1

        target_seq = target_seq.to(loung.device)
        # train a Loung seq2seq model
        loung.zero_grad()
        loung.encoder.zero_grad()
        loung.decoder.zero_grad()
        output_seq_probs, output_seq, hidden, attention, context = loung(batch, target_seq)
        loss = 0
        for i in range(OUTPUT_SIZE):
            loss += loss_function(output_seq_probs[:, i, :], target_seq[:, i])
        loss.backward()
        opitmizer.step()
        losses.append(loss.detach().cpu().item())
        #samples.append((get_sequence_from_indexes(ds.input_word_to_index, batch[0,:]), get_sequence_from_indexes(ds.output_word_to_index, output_seq.squeeze(-1))))
        samples.append((target_seq.detach().cpu().numpy(), output_seq.detach().cpu().numpy()))  

    # plot loss
    plt.figure()
    plt.title("Training")
    plt.plot(losses)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.show(block=False)


################################ Validation #############################

    print('Validation ...')
    for batch, target_seq in validation_dataloader:
        target_seq = target_seq.to(loung.device)
        output_seq_probs, output_seq, hidden, attention, context = loung(batch, target_seq)
        samples.append((target_seq.detach().cpu().numpy(), output_seq.detach().cpu().numpy()))  
    
    with open('validation_results.txt', 'w') as f:
        f.write(str(samples))




 






if __name__ is '__main__':
    main()

# %%
