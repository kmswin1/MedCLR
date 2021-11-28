from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import glob, os
import numpy as np

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

class MedDataset(Dataset):

  def read_data_set(self):

    all_img_files = []
    all_labels = []

    for img_file in self.list_imgs:
      if "test" in img_file:
        label = img_file[22]
      else:
        label = img_file[23]
      
      all_labels.append(label)

    return all_labels

  def __init__(self, list_imgs, train=True ,transform=None):

    self.list_imgs = list_imgs
    # self.labels = np.asarray([-1]*len(self.list_imgs))  # if want idx value : np.asarray()  //여기서 레이블 수정후
    self.labels = self.read_data_set() # get label from img file
    self.transform = transform

  def __len__(self):
    return len(self.list_imgs)

  def __getitem__(self, idx):

    if torch.is_tensor(idx):
      idx = idx.tolist()

    img_path = self.list_imgs[idx]
    target = self.labels[idx]

    img = Image.open(img_path)
    
    if self.transform is not None:
      xi = self.transform(img)
      xj = self.transform(img)
    # return xi, target ## return {xi,xj,self.labels[idx]}
  
    return (xi,xj),target


class Loader(object):
    def __init__(self, file_path, batch_size, train_transform, test_transform, use_cuda):

        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        

        # Get the datasets
        train_dataset, test_dataset = self.get_dataset(MedDataset, file_path,
                                                       train_transform, test_transform)
        # Set the loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        # tmp_batch = self.train_loader.__iter__().__next__()[0]
        # self.img_shape = list(tmp_batch.size())[1:]
        self.num_class = 4

    @staticmethod
    def get_dataset(dataset, file_path, train_transform, test_transform):

        # Training and Validation datasets
        train_dt = glob.glob(os.path.join(file_path,'train/*.jpg'))
        test_dt = glob.glob(os.path.join(file_path,'test/*.jpg'))
        train_dataset = dataset(train_dt, train=True,
                                transform=train_transform)

        test_dataset = dataset(test_dt, train=False,
                               transform=test_transform)

        return train_dataset, test_dataset


def main():

    data_dir = "./data/"
    use_cuda = "cuda:0" if torch.cuda.is_available() else 'cpu'
    print(use_cuda)
    batch_size = 1
    loader = Loader(data_dir, batch_size , train_transform, test_transform, use_cuda)
    train_loader = loader.train_loader
    test_loader = loader.test_loader

    for x, y in train_loader:
        print("x_len:{0}, xi_shape:{1}, xj_shape:{2}, x_type:{3}, y:{4}".format(len(x), x[0].shape,x[1].shape, type(x[0]), y))
    print('-----')

if __name__ == "__main__":
    main()

