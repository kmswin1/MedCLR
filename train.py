import torchvision.models as models
import torch
import torch.nn as nn
from torch.utils.data import dataloader
from dataset import MedDataset

class Trainer:
    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        self.loss = nn.CrossEntropyLoss()

    def train(self, batch_data):

    def test(self, batch_data):


if __name__ == '__main__':
    trainer = Trainer()
    dataset =
