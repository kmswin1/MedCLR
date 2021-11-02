import torchvision.models as models
import torch
import torch.nn as nn
from torch.utils.data import dataset, dataloader

class Tester:
    def __init__(self, model):
        self.model = model
        self.loss = nn.CrossEntropyLoss()

    def train(self):
