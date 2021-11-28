import torchvision.models as models
import torch
import torch.nn as nn
from torch.utils.data import dataloader
from dataset import MedDataset, Loader
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import glob, os
import numpy as np

train_transform = T.Compose([
                    T.Resize((250,250)),
                    T.RandomResizedCrop(224),
                    T.RandomApply([
                            T.ColorJitter(0.5, 0.5, 0.5)
                            ], p=0.8),
                    T.RandomGrayscale(p=0.2),
                    T.ToTensor(),
                    T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

train_transform = T.Compose(
    [
        T.Resize((256,256)),
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=224),
        T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=9),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ]
)

test_transform = T.Compose(
    [
        T.Resize((256,256)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ]
)

class Trainer:
    def __init__(self, out_dim):

        self.model = models.resnet18(pretrained=False, num_classes=out_dim)
        self.loss = nn.CrossEntropyLoss()

    def train(self, batch_data):


    def test(self, batch_data):


if __name__ == '__main__':
    trainer = Trainer(4)
    data_dir = "./data/"
    use_cuda = "cuda:0" if torch.cuda.is_available() else 'cpu'
    print(use_cuda)

    loader = Loader(data_dir, 128, train_transform, test_transform, use_cuda)
    train_loader = loader.train_loader
    test_loader = loader.test_loader

    for i,
