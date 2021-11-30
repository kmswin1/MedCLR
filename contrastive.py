import torchvision.models as models
import torch
import torch.nn as nn
from torch.utils.data import dataset, dataloader

import logging
import os
import sys
from dataset import MedDataset, Loader
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from model import ResNetSimCLR
from torchvision import transforms as T


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

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self):
        self.temperature = 0.2
        self.model = ResNetSimCLR(4).to('cuda:0')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        self.criterion = torch.nn.CrossEntropyLoss().to('cuda:0')

    def info_nce_loss(self, features, batch_size):

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to('cuda:0')

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to('cuda:0')
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to('cuda:0')

        logits = logits / self.temperature
        return logits, labels

    def train(self, train_loader):

        # save config file
        n_iter = 0
        optim_loss = 9999
        with open('pretrain_log.txt', 'w') as f:
            for epoch_counter in range(1000):
                tot=0
                train_loss = 0
                for images, _ in tqdm(train_loader):
                    images = torch.cat(images, dim=0).to('cuda:0')
                    batch_size = int(len(images)/2)
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features, batch_size)
                    loss = self.criterion(logits, labels)

                    self.optimizer.zero_grad()

                    loss.backward()

                    self.optimizer.step()

                    train_loss = loss.item()
                    tot += batch_size
                print("train loss : " + str(train_loss))
                f.write('train_loss : ' + str(train_loss) + '\n')
                if optim_loss > train_loss:
                    print ("model saved...")
                    optim_loss = train_loss
                    torch.save(self.model.state_dict(), 'model.pt')

if __name__ == '__main__':
    trainer = SimCLR()
    data_dir = "./data/"
    use_cuda = "cuda:0" if torch.cuda.is_available() else 'cpu'
    print(use_cuda)
    optimal_accuracy = 0

    loader = Loader(data_dir, 128, train_transform, test_transform, use_cuda)
    train_loader = loader.train_loader
    test_loader = loader.test_loader
    tot = 0
    loss = 0
    trainer.train(train_loader)