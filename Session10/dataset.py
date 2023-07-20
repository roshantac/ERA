##########################---Dataset--#############################
import torch
import torchvision
import torchvision.transforms as transforms
#import Albumentation

import numpy as np
from albumentations import Compose, RandomCrop,PadIfNeeded, Normalize, HorizontalFlip,HueSaturationValue,Cutout,ShiftScaleRotate
import albumentations as A
from albumentations.pytorch import ToTensorV2
#from albumentations.pytorch import ToTensor
import cv2


class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
  def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
      super().__init__(root=root, train=train, download=download, transform=transform)

  def __getitem__(self, index):
      image, label = self.data[index], self.targets[index]
      if self.transform is not None:
          transformed = self.transform(image=image)
          image = transformed["image"]
      return image, label


def get_data_alb():

  train_transform = A.Compose([
      A.PadIfNeeded(min_height=32, min_width=32, border_mode=cv2.BORDER_CONSTANT, value=4, always_apply=False, p=.50),
      A.RandomCrop(32, 32, always_apply=False, p=.50),A.HorizontalFlip(),
      A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=13, val_shift_limit=9),
      A.Cutout(num_holes=1, max_h_size=8, max_w_size=8),
      A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2,rotate_limit=13, p=0.6),

      A.Normalize(
        mean = [0.4914, 0.4822, 0.4465],
        std = [0.2470, 0.2435, 0.2616],
        ),
      ToTensorV2(),
      ],p=1)
  train_data = Cifar10SearchDataset(root='./data', train=True,download=True, transform=train_transform)
  trainloader = torch.utils.data.DataLoader(train_data, batch_size= 512,
                                            shuffle=True, num_workers=2)
  test_transforms = A.Compose(

    [ A.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2470, 0.2435, 0.2616),p =1.0),ToTensorV2()], p=1.0)

  test_data = Cifar10SearchDataset(root='./data', train=False,download=True, transform=test_transforms)
  testloader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=False, num_workers=2)
  testset2 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
  testloader2 = torch.utils.data.DataLoader(testset2, batch_size=1, shuffle=False, num_workers=2)
  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  return trainloader, testloader,testloader2, classes
class album_Compose:
  def __init__(self):
    self.alb_transform =A.Compose([
      A.PadIfNeeded(min_height=32, min_width=32, border_mode=cv2.BORDER_CONSTANT, value=4, always_apply=False, p=.50),
      A.RandomCrop(32, 32, always_apply=False, p=.50),A.HorizontalFlip(),
      A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=13, val_shift_limit=9),
      A.Cutout(num_holes=1, max_h_size=8, max_w_size=8),
      A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2,rotate_limit=13, p=0.6),

      A.Normalize(
        mean = [0.4914, 0.4822, 0.4465],
        std = [0.2023, 0.1994, 0.2010],
        ),
      ToTensorV2(),
      ],p=.8)
  def __call__(self,img):
    img = np.array(img)
    img = self.alb_transform(image=img)['image']
    return img



def getData():
  k= album_Compose()
  transform_test = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
  transform_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
      transforms.RandomRotation((-10.0, 10.0)), transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform_train)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size= 512,
                                            shuffle=True, num_workers=2)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                          shuffle=False, num_workers=2)
  testset2 = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transforms.Compose([transforms.ToTensor()]))
  testloader2 = torch.utils.data.DataLoader(testset2, batch_size=1,
                                          shuffle=False, num_workers=2)

  classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  return trainloader, testloader,testloader2, classes
