from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import glob, os
import numpy as np

class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = T.ToTensor()
        self.tensor_to_pil = T.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class MedDataset(Dataset):

  def __init__(self, list_imgs, train=True ,transform=None):

    self.list_imgs = list_imgs
    self.labels = np.asarray([-1]*len(self.list_imgs))
    self. transform = transform

  def __len__(self):
    return len(self.list_imgs)

  def __getitem__(self, idx):

    if torch.is_tensor(idx):
      idx = idx.tolist()

    img_path = self.list_imgs[idx]
    target = self.labels
    img = Image.open(img_path)
    
    if self.transform is not None:
      xi = self.transform(img)
    #   xj = self.transform(img)

    return xi, target

class Loader(object):
    def __init__(self, file_path, batch_size, train_transform, test_transform, use_cuda):

        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

        

        # Get the datasets
        train_dataset, test_dataset = self.get_dataset(MedDataset, file_path,
                                                       train_transform, test_transform)
        # Set the loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        tmp_batch = self.train_loader.__iter__().__next__()[0]
        self.img_shape = list(tmp_batch.size())[1:]
        self.num_class = 4

    @staticmethod
    def get_dataset(dataset, file_path, train_transform, test_transform):

        # Training and Validation datasets
        train_dataset = dataset(file_path, train=True,
                                transform=train_transform)

        test_dataset = dataset(file_path, train=False,
                               transform=test_transform)

        return train_dataset, test_dataset


def main():

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
    data_dir = glob.glob("C:/Users/SINGKU/Desktop/data/*.jpg")
    # data_dir = glob.glob("./data/*.jpg")
    use_cuda = "cuda:0" if torch.cuda.is_available() else 'cpu'
    print(use_cuda)

    loader = Loader(data_dir, 4 , train_transform, train_transform, use_cuda)
    train_loader = loader.train_loader
    test_loader = loader.test_loader

    for x, y in train_loader:
        print("x_len:{0}, x_shape:{1}, x_type:{2}, y:{3}".format(len(x), x[0].shape, type(x[0]), y))
    print('-----')

if __name__ == "__main__":
    main()