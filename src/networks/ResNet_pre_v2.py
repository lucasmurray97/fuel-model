import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
import sys
from matplotlib import pyplot as plt
from networks.ae import AutoEncoder

class ResNet_Pre_18_V2(nn.Module):
    def __init__(self, params = {}):
        super(ResNet_Pre_18_V2, self).__init__()
        self.name = "Res-Net_Pre_18_V2"
        # Loading ResNet18 and modify first convolution to fit our input
        self.base_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # Classifier
        self.fc_1 = nn.Linear(in_features=1000, out_features=256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc_2 = nn.Linear(in_features=256, out_features=20)
        # Encoder layers:
        self.conv1_ = nn.ConvTranspose2d(in_channels=71, out_channels=64, kernel_size=(4,4), stride=3)
        self.conv2_ = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4,4), stride=2)
        self.conv3_ = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(2,2), stride=2)
        self.conv4_ = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=(2,2), stride=2)
        self.encoder = nn.Sequential(
          self.conv1_,
          nn.ReLU(),
          self.conv2_,
          nn.ReLU(),
          self.conv3_,
          nn.ReLU(),
          self.conv4_,
          nn.ReLU(),
        )
        # Loss func
        self.softmax = torch.nn.Softmax(dim=1) 
        self.loss_func = nn.CrossEntropyLoss()
        self.training_loss = []
        self.epoch_loss = 0
        self.n = 0
        self.validation_loss = []
        self.val_epoch_loss = 0
        self.m = 0
        self.augment = params["augment"]

    def forward(self, x):
        #print(x.shape)
        x = self.encoder(x)
        #print(x.shape)
        x = self.base_model(x)
        #print(x.shape)
        x = F.relu(self.bn1(self.fc_1(x)))
        #print(x.shape)
        x = self.fc_2(x)
        #print(x.shape)
        return x
    
    
    def train_loss(self, pred, y):
        loss = self.loss_func(pred, y.squeeze(1))
        self.epoch_loss += loss.item()
        self.n += 1
        return loss

    def val_loss(self, pred, y):
        loss = self.loss_func(pred, y.squeeze(1))
        self.val_epoch_loss += loss.item()
        self.m += 1
        return loss
    
    def reset_losses(self):
        self.training_loss.append(self.epoch_loss / self.n)
        self.epoch_loss = 0
        self.n = 0
        self.validation_loss.append(self.val_epoch_loss / self.m)
        self.val_epoch_loss = 0
        self.m = 0
    
    def plot_loss(self, epochs, stage):
        self.to("cpu")
        plt.ion()
        fig = plt.figure()
        plt.plot(self.training_loss, label='training loss')
        plt.plot(self.validation_loss, label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"./plots/Losses_{stage}_{self.name}_{epochs}_{self.augment}.png")

    def finish_1(self, epochs):
        self.plot_loss(epochs, "stage_1")
        self.training_loss = []
        self.validation_loss = []

    def finish_2(self, epochs):
        self.plot_loss(epochs, "stage_2")
        path_ = f"./networks/weights/{self.name}_{epochs}_{self.augment}.pth"
        torch.save(self.state_dict(), path_)