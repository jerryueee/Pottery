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

available_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resolution = 64
z_latent_space = 128
batch_size = 64
checkpoint_dir = './model_path/mode.path'
theta = 0.1

def criterion(outputs, targets):
    return nn.MSELoss()(outputs, targets)

def criterion_dsc(outputs, targets):
    batch_size = outputs.size(0)
    dsc_list = []
    for i in range(batch_size):
        output = outputs[i].view(-1)
        target = targets[i].view(-1)
        dsc = 2 * (output * target).sum() / (output.sum() + target.sum())
        dsc_list.append(1-dsc)
    return torch.tensor(dsc_list)

def criterion_jd(outputs, targets):
    # jaccard = (outputs * targets).sum() / (outputs + targets).sum()
    batch_size = outputs.size(0)
    dsc_list = []
    for i in range(batch_size):
        output = outputs[i].view(-1)
        target = targets[i].view(-1)
        com =(output * target).sum()
        dsc = com / (output.sum() + target.sum()- com)
        dsc_list.append(1-dsc)
    return torch.tensor(dsc_list)

def test(model, test_ratio,use_load):
    # TODO
    # You can also implement this function in training procedure, but be sure to
    # evaluate the model on test set and reserve the option to save both quantitative
    # and qualitative (generated .vox or visualizations) images.

    # G = Generator(cube_len=resolution, z_latent_space=z_latent_space, z_intern_space=resolution).to(available_device)       
    if use_load:
        model.load_state_dict(torch.load(checkpoint_dir,weights_only=True))   
    testset = FragmentDataset('data', 'test', dim_size=resolution)
    test_size = int(len(testset) * test_ratio)
    testset = data.random_split(testset, [test_size, len(testset) - test_size])[0]
    testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    # test
    correct=0
    total=0
    with torch.no_grad():
        for frags, voxels in tqdm(testloader):
            frags, voxels = frags.to(available_device), voxels.to(available_device)
            outputs = model(frags)
            losses = criterion(outputs, voxels)
            correct += (losses < theta).sum().item()
            total += voxels.size(0)           
    print(f'Accuracy of pottery generated: {correct / total}')
