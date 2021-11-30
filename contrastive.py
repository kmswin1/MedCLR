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
from utils import save_config_file, accuracy, save_checkpoint
from model import ResNetSimCLR
from utils import EarlyStopping
from torchvision import transforms as T

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

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.temperature = 0.2
        self.model = ResNetSimCLR(4).to('cuda:0')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to('cuda:0')

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(128) for i in range(2)], dim=0)
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

        for epoch_counter in range(100):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to('cuda:0')

                features = self.model(images)
                logits, labels = self.info_nce_loss(features)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

if __name__ == '__main__':
    trainer = SimCLR(4)
    early_stopping = EarlyStopping()
    data_dir = "./data/"
    use_cuda = "cuda:0" if torch.cuda.is_available() else 'cpu'
    print(use_cuda)
    optimal_accuracy = 0

    loader = Loader(data_dir, 128, train_transform, test_transform, use_cuda)
    train_loader = loader.train_loader
    test_loader = loader.test_loader
    with open('log.txt', 'w') as f:
        for epoch in range(1, 100):
            tot = 0
            loss = 0
            for i, batch_data in enumerate(train_loader):
                loss += trainer.train(batch_data)
                tot += batch_data[1].size(0)

            print ("train loss : " + str(loss))
            f.write("train loss : " + str(loss) + '\n')

            #tot = 0
            #loss = 0
            #for i, batch_data in enumerate(test_loader):
            #    valid_loss, accuracy = trainer.test(batch_data)
            #    loss += valid_loss
            #    tot += batch_data[1].size(0)

            #print("valid loss : " + str(loss))
            #print("Accuracy : "+ str(accuracy))

            #f.write("valid loss : " + str(loss) + '\n')
            #f.write("Accuracy : " + str(accuracy) + '\n')




            #if accuracy > optimal_accuracy:
            #    optimal_accuracy = accuracy
            #    print ("model saved")
            #    torch.save(trainer.model, 'model.pt')

            if early_stopping(loss):
                print ("early stopped ...")
                if accuracy > optimal_accuracy:
                    torch.save(trainer.model, 'model.pt')