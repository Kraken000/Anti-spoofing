from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F

import os


BS = 32
# Image transformations



# Train uses data augmentation
image_train_transforms = transforms.Compose([
        # transforms.Resize(size=64),
        transforms.RandomResizedCrop(size=112, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])


image_valid_transforms = transforms.Compose([
        transforms.Resize(size=112),
        transforms.CenterCrop(size=64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])



traindir = './dataset/train/'
# Datasets from folders
data_train = datasets.ImageFolder(root='./dataset/train/', transform=image_train_transforms)

# print(data_train.class_to_idx)
# print(data_train.classes)

# trainiter = iter(dataloaders_train)
# img, labels = next(trainiter)
# print(img.shape, labels.shape)

dataloaders_train = torch.utils.data.DataLoader(data_train, batch_size=32, shuffle=True, num_workers=2)

cirterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters())




for i, data in enumerate(dataloaders_train, 0):
    img, lbl = data

    print(i, "inputs", img.data.size(), "labels", lbl.data.size())



class NewDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):

        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]