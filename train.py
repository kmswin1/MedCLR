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
from utils import EarlyStopping

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, batch_data):
        self.optimizer.zero_grad()
        x_i, x_j, labels = batch_data
        pred = self.model(x_i)
        loss = self.loss(pred, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def test(self, batch_data):
        with torch.no_grad():
            x_i, x_j, labels = batch_data
            labels = torch.tensor(labels, dtype=torch.int64, device='cuda:0')
            pred = self.model(x_i)
            loss = self.loss(pred, labels)
            pred = torch.argmax(pred, dim=-1)
            accuracy = torch.sum(torch.eq(pred, labels))
            accuracy /= pred.size(0)


if __name__ == '__main__':
    trainer = Trainer(4)
    early_stopping = EarlyStopping()
    data_dir = "./data/"
    use_cuda = "cuda:0" if torch.cuda.is_available() else 'cpu'
    print(use_cuda)
    optimal_accuracy = 0

    loader = Loader(data_dir, 128, train_transform, test_transform, use_cuda)
    train_loader = loader.train_loader
    test_loader = loader.test_loader
    tot = 0
    loss = 0
    with open('log.txt', 'w') as f:
        for i, batch_data in enumerate(train_loader):
            loss += Trainer.train(batch_data)
            tot += batch_data[0].size(0)

        print ("train loss : " + str(loss))
        f.write("train loss : " + str(loss) + '\n')

        tot = 0
        loss = 0
        for i, batch_data in enumerate(test_loader):
            valid_loss, accuracy = Trainer.test(batch_data)
            loss += valid_loss
            tot += batch_data[0].size(0)

        print("valid loss : " + str(loss))
        print("Accuracy : "+ str(accuracy))

        f.write("valid loss : " + str(loss) + '\n')
        f.write("Accuracy : " + str(accuracy) + '\n')




        if accuracy > optimal_accuracy:
            print ("model saved")
            torch.save(trainer.model, 'model.pt')

        if early_stopping(loss):
            print ("early stopped ...")
            if accuracy > optimal_accuracy:
                torch.save(trainer.model, 'model.pt')