# Sequence to sequence with attention (seq2seq-with-deep-attention)
Sequence to sequence (seq2seq) using RNN with attention mechanism in pytorch.

This work is the pytorch implementation of the Loung seq2seq model https://arxiv.org/pdf/1508.04025.pdf. 

We have used the date generation code from https://github.com/tensorflow/tfjs-examples/tree/master/date-conversion-attention to create a dataset and have followed the same steps in this link, but our implementation uses NLLoss instead of cat. softmax loss function. 
The current implementation is meant for learninig purposes and might not be effecient in terms of speed. However, it can show the effeciencey of this model even when trained for a small number of steps.


## To run:
  see `example.ipynb`
## For TensorFlow impelementation and more useful links:
please check: https://github.com/AndreMaz/deep-attention


