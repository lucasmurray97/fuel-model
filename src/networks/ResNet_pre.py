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

class ResNet_Pre_18(nn.Module):
    def __init__(self, params = {}):
        super(ResNet_Pre_18, self).__init__()
        self.name = "Res-Net_Pre_18"
        # Loading ResNet18 and modify first convolution to fit our input
        self.base_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # Classifier
        self.fc_1 = nn.Linear(in_features=1000, out_features=256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc_2 = nn.Linear(in_features=256, out_features=20)
        # Encoder
        self.encoder = AutoEncoder()
        self.encoder.load_state_dict(torch.load("./networks/weights/AE_100.pth"))
        for param in self.encoder.parameters():
            param.requires_grad = False
        # Loss func
        self.softmax = torch.nn.Softmax(dim=1) 
        self.loss_func = nn.CrossEntropyLoss()
        self.training_loss = []
        self.epoch_loss = 0
        self.n = 0
        self.validation_loss = []
        self.val_epoch_loss = 0
        self.m = 0

    def forward(self, x):
        #print(x.shape)
        x = self.encoder.encode(x)
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