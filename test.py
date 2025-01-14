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
from tqdm import tqdm
import utils.visualize as vis

available_device = torch.device('cpu')
resolution = 64
z_latent_space = 128
batch_size = 64
checkpoint_dir = './model_path/model.path'
theta = 0.1


JD_threshold = 0.9
DCS_threshold = 0.1
MSE_threshold = 0.1

def criterion(outputs, targets):
    return nn.MSELoss()(outputs, targets)

def criterion_dcs(outputs, targets):
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

def JD(fake:torch.Tensor, real:torch.Tensor):
    fake = fake.bool()
    real = real.bool()
    intersection = fake & real
    union = fake | real
    si = torch.sum(intersection, dim=(1, 2, 3))
    su = torch.sum(union, dim=(1, 2, 3))
    jd = (su - si) / su
    return jd

def DCS(fake:torch.Tensor, real:torch.Tensor):
    fake = fake.bool()
    real = real.bool()
    intersection = fake & real
    si = torch.sum(intersection, dim=(1, 2, 3))
    sa = torch.sum(fake, dim=(1, 2, 3))
    sb = torch.sum(real, dim=(1, 2, 3))
    dsc = 2 * si / (sa + sb)
    return dsc

def MSE(fake:torch.Tensor, real:torch.Tensor):
    fake = fake>0.5
    real = real
    return nn.MSELoss()(fake, real)

def test():
    # TODO
    # You can also implement this function in training procedure, but be sure to
    # evaluate the model on test set and reserve the option to save both quantitative
    # and qualitative (generated .vox or visualizations) images.

    # G = Generator(cube_len=resolution, z_latent_space=z_latent_space, z_intern_space=resolution).to(available_device)       
    model = Generator(cube_len=resolution, z_latent_space=z_latent_space, z_intern_space=resolution).to(available_device)
    model.load_state_dict(torch.load(checkpoint_dir, weights_only=True))
    # ? weights_only = True
    model.eval()
    testset = FragmentDataset('data', 'test', dim_size=resolution)
    testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    # test
    correct = 0
    total = 0
    cnt = 0
    with torch.no_grad():
        for frags, voxels in tqdm(testloader, desc=f"epoch:{1}/{1}", unit="batch"):
            frags, voxels = frags.to(available_device), voxels.to(available_device)
            outputs = model(frags)
            if total  % batch_size == 0:
                # print(outputs[0].size())
                # print(outputs[0])
                # vis.plot_frag(np.ones((32,32,32)), 'test.png')
                # vis.plot_frag(np.array(outputs[0]), 'test.png')
                index = np.random.randint(len(outputs))
                vis.plot_join(np.array(outputs[index]),frags[index], f'test_result{cnt+1}.png')
                cnt += 1
            # if total % batch_size == 0 and total > 8 * batch_size:
            #     i = np.random.randint(len(outputs))
            #     vis.plot_join(np.array(outputs[i]),frags[i], 'test_join.png')
            losses = MSE(outputs, voxels)
            correct += (losses).sum().item()
            total += voxels.size(0)           
    print(f'loss of pottery generated: {correct / total}')
