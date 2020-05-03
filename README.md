# Sequence to sequence with attention (seq2seq-with-deep-attention)

This work is the minimal pytorch implementation of some sequence to sequence models.

## Keywords:
sequence to sequence ; Loung; NLP; optimization; Pointer networks


## Settings:
- pytorch 1.4.0
- python 3.7.6


## 1- Loung seq2seq model.
 We have used the date generation code from https://github.com/tensorflow/tfjs-examples/tree/master/date-conversion-attention to create a dataset and have followed the same steps in this link, but our implementation uses NLLoss instead of cat. softmax loss function. 
The current implementation is meant for learninig purposes and might not be effecient in terms of speed.

### To run:
  see `Loung_seq2seq_example.ipynb`

### For TensorFlow impelementation and more useful links:
  please check: https://github.com/AndreMaz/deep-attention


## 2- Pointer nets.
 This is simillar to the previouse model but in this case we use the softmax output to point back to the input. Thus making the output and input length consistent and removes the limitation in predefining a fixed output length. For more details see the https://arxiv.org/abs/1506.03134

### sorting numbers using pointer nets
 I will use number sorting example to demonstrate how pointer network works.

 Let us assume a pointer net is a blackbox machine, where we feed it with an array (sequence) of unsorted numbers. The machine sorts the array. We are sure there is not any sorting algorithm inside the machine. How does it do it?


![png](images/ptr_machine_1.png=100x20)
  
 ### To run:
  - see `pointer_net_example.ipynb` for unmasked pointer network 
  - see  `masked_pointer_net_example.ipynb` for masked pointer network, notice the radically improved performance!
  - see  `pointer_net_multi_features_example.ipynb` for masked pointer network used for input vector with higher dimensions.

