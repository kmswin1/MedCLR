import torchvision.models as models
import torch
import torch.nn as nn
from torch.utils.data import dataset, dataloader

class Trainer:
    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        self.loss = nn.CrossEntropyLoss()

    def train(self):
