
#%%  main runner file
import torch

import sys
sys.path.append("..")

import seq2seq_with_deep_attention.runners.run_loung as run_loung
import seq2seq_with_deep_attention.runners.run_pointer_net as run_pointer_net
import seq2seq_with_deep_attention.runners.run_masked_pointer_net as run_masked_pointer_net
import seq2seq_with_deep_attention.runners.run_pointer_nets_multi_features as run_pointer_nets_multi_features

random_seed = torch.manual_seed(45)

def main():
    # uncommnet a line to run an example
    #run_loung.main()
    #run_pointer_net.main()
    run_masked_pointer_net.main()
    #run_pointer_nets_multi_features.main()



if __name__ is '__main__':
    main()

# %%
