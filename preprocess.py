import torch
import torchvision.transforms as transforms
import random

def get_transform(name='cifar10', img_size=None):
    if name=='cifar10':
        image_size = img_size or 64
        return transforms.Compose([transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0,0,0), (1,1,1)),])
