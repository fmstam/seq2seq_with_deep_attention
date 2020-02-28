# main run file
import torch

import seq2seq_with_deep_attention.run_loung as run_loung
import seq2seq_with_deep_attention.run_pointer_net as run_pointer_net

random_seed = torch.manual_seed(45)

def main():
    #run_loung.main()
    run_pointer_net.main()


if __name__ is '__main__':
    main()