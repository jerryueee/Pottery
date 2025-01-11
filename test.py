## Here you may implement the evaluation method and may call some necessary modules from utils.model_utils.py
## Derive the test function by yourself and implement proper metric such as Dice similarity coeffcient (DSC)[4];
# Jaccard distance[5] and Mean squared error (MSE), etc. following the handout in model_utilss.py

import numpy as np
import torch
from torch import optim
from torch.utils import data
from torch import nn
from utils.FragmentDataset import FragmentDataset
from utils.model import Generator, Discriminator
import tqdm
import training

def test():
    # TODO
    # You can also implement this function in training procedure, but be sure to
    # evaluate the model on test set and reserve the option to save both quantitative
    # and qualitative (generated .vox or visualizations) images.       
    G.load_state_dict(torch.load(checkpoint_dir,weights_only=True))

    
    testset = FragmentDataset(dirdataset, 'test', dim_size=resolution)
    testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    # test
    correct=0
    total=0
    with torch.no_grad():
        for frags, voxels in tqdm(testloader):
            frags, voxels = frags.to(available_device), voxels.to(available_device)
            outputs = G(frags)
            correct += (Gcriterion3(outputs, voxels)).sum().item
            

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')
