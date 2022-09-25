from __future__ import print_function
import argparse
import os
import logging
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import models
# import torchvision.models as models

# @azatkariuly
from preprocess import get_transform
from data import get_dataset
from datetime import datetime

from torch.nn.functional import adaptive_avg_pool2d

from PIL import Image

import sys
import numpy as np
import os
# print(os.listdir("../input"))

import time

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch GAN Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='dcgan',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--gen_lr', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate of generator')
parser.add_argument('--dis_lr', default=4e-4, type=float,
                    metavar='LR', help='initial learning rate of discriminator')

SEED=42
random.seed(SEED)
torch.manual_seed(SEED)
# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 64
# Number of training epochs
num_epochs = 70
# different Learning rate for optimizers
g_lr = 0.0001
d_lr = 0.0004
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
ngpu=1

def main():
    global args, best_res
    best_res = 0
    args = parser.parse_args()


    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    model_config = {'dataset': args.dataset}

    netG, netD = model(**model_config)
    logging.info("created model with configuration: %s", model_config)

    # Data loading code
    # transform = get_transform(args.dataset)
    # dataset = get_dataset(args.dataset, transform)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
    #                                          shuffle=True, num_workers=2)

    transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0,0,0), (1,1,1)),])

    dataset = dset.CIFAR10(root='../data2', train=True,
                                        download=True, transform=transform)
    print(dataset)
    return

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2)

    for i, data in enumerate(dataloader, 0):
        print('hbeajk')
        print(i, data)
        return

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=args.dis_lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netD.parameters(), lr=args.gen_lr, betas=(0.5, 0.999))

    img_list = []
    G_losses = []
    D_losses = []
    iters = []

    print('Training..')
    for epoch in range(args.epochs):
        for i, data in enumerate(dataloader, 0):

            print(i, data)
            '''
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
    #         # add some noise to the input to discriminator

            real_cpu=0.9*real_cpu+0.1*torch.randn((real_cpu.size()), device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)

            fake=0.9*fake+0.1*torch.randn((fake.size()), device=device)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            D_G_z2 = output.mean().item()

            # Calculate gradients for G
            errG.backward()
            # Update G
            optimizerG.step()
            if i%100 == 0:
                print('[%d/%d]\t iteration %d/%d'
                              % (epoch+1, num_epochs, i, len(dataloader)))
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fixed_noise = torch.randn(ngf, nz, 1, 1, device=device)
                    fake_display = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake_display, padding=2, normalize=True))



            iters += 1
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        fretchet_dist=0 #calculate_fretchet(real_cpu,fake,model)
        # if ((epoch+1)%5==0):

        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tFretchet_Distance: %.4f'
                      % (epoch+1, num_epochs,
                          errD.item(), errG.item(),fretchet_dist))
'''
if __name__ == '__main__':
    main()
