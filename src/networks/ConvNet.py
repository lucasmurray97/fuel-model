import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

class ConvNet(nn.Module):
    def __init__(self, params = {}):
        super(ConvNet, self).__init__()
        self.name = "Conv-Net"
        self.conv1 = nn.Conv2d(in_channels=71, out_channels=16, kernel_size=(4,4), stride=(4,4))
        self.bn1 = nn.BatchNorm2d(16)
        self.max_pool_1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3)) # (64, 10, 10)
        self.bn2 = nn.BatchNorm2d(32)
        self.max_pool_2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2,2)) # (64, 10, 10)
        self.bn3 = nn.BatchNorm2d(32)
        self.max_pool_3 = nn.MaxPool2d(2)
        self.fc_1 = nn.Linear(in_features=512, out_features=256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc_2 = nn.Linear(in_features=256, out_features=20)
        self.softmax = torch.nn.Softmax(dim=1) 
        self.loss_func = nn.CrossEntropyLoss()
        self.training_loss = []
        self.epoch_loss = 0
        self.n = 0
        self.validation_loss = []
        self.val_epoch_loss = 0
        self.m = 0

    def forward(self, x):
        x = self.max_pool_1(F.relu(self.bn1(self.conv1(x))))
        #print(x.shape)
        x = self.max_pool_2(F.relu(self.bn2(self.conv2(x))))
        #print(x.shape)
        x = self.max_pool_3(F.relu(self.bn3(self.conv3(x))))
        #print(x.shape)
        x = torch.flatten(x, start_dim=1)
        #print(x.shape)
        x = F.relu(self.bn4(self.fc_1(x)))
        x = self.fc_2(x)
        return x
    
    def train_loss(self, pred, y):
        target = torch.argmax(y, dim=1).long()
        loss = self.loss_func(pred, target)
        self.epoch_loss += loss.item()
        self.n += 1
        return loss

    def val_loss(self, pred, y):
        target = torch.argmax(y, dim=1)
        loss = self.loss_func(pred, target)
        self.val_epoch_loss += loss.item()
        self.m += 1
        return loss
    
    def reset_losses(self):
        self.training_loss.append(self.epoch_loss / self.n)
        self.epoch_loss = 0
        self.n = 0
        #self.validation_loss.append(self.val_epoch_loss / self.m)
        #self.val_epoch_loss = 0
        #self.m = 0
    
    def plot_loss(self, epochs):
        self.to("cpu")
        plt.ion()
        fig = plt.figure()
        plt.plot(self.training_loss, label='training loss')
        plt.plot(self.validation_loss, label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"./plots/Losses_{self.name}_{epochs}.png")

    def finish(self, epochs):
        self.plot_loss(epochs)
        path_ = f"./networks/weights/{self.name}_{epochs}.pth"
        torch.save(self.state_dict(), path_)