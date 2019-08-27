import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import os

class Module1(nn.Module):
    def __init__(self):
        super(Module1, self).__init__()

        '''# first CONV => RELU => CONV => RELU => POOL layer set
        
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.bn1_1 = nn.BatchNorm2d(16)
        self.conv1_2 = nn.Conv2d(16, 16, 3)
        self.bn1_2 = nn.BatchNorm2d(16)
        self.maxpool1 = nn.MaxPool2d(2)

        # second CONV => RELU => CONV => RELU => POOL layer set
        self.conv2_1 = nn.Conv2d(16, 32, 3)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 32, 3)
        self.bn2_2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(2)
        '''

        # first CONV => RELU => CONV => RELU => POOL layer set
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),

            # second CONV => RELU => CONV => RELU => POOL layer set
            nn.Conv2d(16, 32, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )
        # first (and only) set of FC => RELU layers
        # self.fc = nn.Sequential(
        #     nn.Linear()
        # )

        # self.fc = nn.Sequential(
        #
        # )
    def forward(self, x):
        x =self.conv(x)

        # print(x.shape)
        x = x.view(x.shape[0], -1)
        # print(x.shape[1])

        self.fc1 = nn.Linear(x.shape[1], 64)
        self.fc2 = nn.Linear(64, 2)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x

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

dataloaders_train = torch.utils.data.DataLoader(data_train, batch_size=2, shuffle=True, num_workers=2)

if __name__ == '__main__':
    net = Module1()
    cirterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    '''print(net)
    print(net.parameters())
    data_input = torch.randn([1, 3, 64, 64])



    print(data_input.size())
    print(net(data_input))'''
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(dataloaders_train):
            img, lbl = data

            optimizer.zero_grad()
            outputs = net(img)
            loss = cirterion(outputs, lbl)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 50 == 49:
                print('[%d %5d loss: %.3f'%(epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
