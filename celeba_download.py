import time
import os
import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

def get_dataloaders_celeba(batch_size, num_workers=0,
                           train_transforms=None,
                           test_transforms=None,
                           download=True):
    """Targets are 40-dim vectors representing
    00 - 5_o_Clock_Shadow
    01 - Arched_Eyebrows
    02 - Attractive
    03 - Bags_Under_Eyes
    04 - Bald
    05 - Bangs
    06 - Big_Lips
    07 - Big_Nose
    08 - Black_Hair
    09 - Blond_Hair
    10 - Blurry
    11 - Brown_Hair
    12 - Bushy_Eyebrows
    13 - Chubby
    14 - Double_Chin
    15 - Eyeglasses
    16 - Goatee
    17 - Gray_Hair
    18 - Heavy_Makeup
    19 - High_Cheekbones
    20 - Male
    21 - Mouth_Slightly_Open
    22 - Mustache
    23 - Narrow_Eyes
    24 - No_Beard
    25 - Oval_Face
    26 - Pale_Skin
    27 - Pointy_Nose
    28 - Receding_Hairline
    29 - Rosy_Cheeks
    30 - Sideburns
    31 - Smiling
    32 - Straight_Hair
    33 - Wavy_Hair
    34 - Wearing_Earrings
    35 - Wearing_Hat
    36 - Wearing_Lipstick
    37 - Wearing_Necklace
    38 - Wearing_Necktie
    39 - Young
    """

    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = datasets.CelebA(root='../data',
                                    split='train',
                                    transform=train_transforms,
                                    download=download)

    valid_dataset = datasets.CelebA(root='../data',
                                    split='valid',
                                    transform=test_transforms)

    test_dataset = datasets.CelebA(root='../data',
                                   split='test',
                                   transform=test_transforms)


    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)

    valid_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    return train_loader, valid_loader, test_loader

custom_transforms = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(64),
    torchvision.transforms.Resize(64),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


train_loader, valid_loader, test_loader = get_dataloaders_celeba(
    batch_size=128,
    train_transforms=custom_transforms,
    test_transforms=custom_transforms,
    num_workers=4)
