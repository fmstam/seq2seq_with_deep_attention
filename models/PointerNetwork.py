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

class PointerNet(nn.Module):
    def __init__(self,
                 )
