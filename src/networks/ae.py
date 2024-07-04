import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.name = "AE"
        # Encoder layers:
        self.conv1_ = nn.ConvTranspose2d(in_channels=71, out_channels=64, kernel_size=(4,4), stride=3)
        self.conv2_ = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4,4), stride=2)
        self.conv3_ = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(2,2), stride=2)
        self.conv4_ = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=(2,2), stride=2)
        
        # Decoder layers:
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2,2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2), stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4,4), stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=71, kernel_size=(4,4), stride=3)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.MSELoss()
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Inicialización de parámetros:
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv1_.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2_.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv3_.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv4_.weight, mode='fan_in', nonlinearity='relu')
        # Optimizer:
        self.optimizer = torch.optim.Adam(self.parameters())
        # Losses holders:
        self.training_loss = []
        self.validation_loss = []
        self.m = 0
        self.n = 0
        self.epoch_loss = 0
        self.val_epoch_loss = 0

    def encode(self, x):
        u1 = F.relu(self.conv1_(x))
        #print(u1.shape)
        u2 = F.relu(self.conv2_(u1))
        #print(u2.shape)
        u3 = F.relu(self.conv3_(u2))
        #print(u3.shape)
        output = F.relu(self.conv4_(u3))
        #print(output.shape)
        return output

    
    def decode(self, x):
        u1 = F.relu(self.conv1(x))
        #print(u1.shape)
        u2 = F.relu(self.conv2(u1))
        #print(u2.shape)
        u3 = F.relu(self.conv3(u2))
        #print(u3.shape)
        output = F.relu(self.conv4(u3))
        #print(output.shape)
        return output
        
    def forward(self, x):
        return self.decode(self.encode(x))
    
    def loss(self, output, x):
        loss = self.criterion(output, x)
        self.epoch_loss += loss.item()
        self.n += 1
        return loss
    
    def val_loss(self, output, x):
        loss = self.criterion(output, x)
        self.val_epoch_loss += loss.item()
        self.m += 1
        return loss
    
    def step(self):
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def reset_losses(self):
        self.training_loss.append(self.epoch_loss/self.n)
        self.validation_loss.append(self.val_epoch_loss/self.m)
        self.epoch_loss = 0
        self.val_epoch_loss = 0
        self.m = 0
        self.n = 0
    
    def plot_loss(self, epochs):
        self.to("cpu")
        plt.ion()
        fig = plt.figure()
        plt.plot(self.training_loss, label='training loss')
        plt.plot(self.validation_loss, label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"../plots/Losses_{self.name}_{epochs}.png")

    def finish(self, epochs):
        self.plot_loss(epochs)
        path_ = f"./weights/{self.name}_{epochs}.pth"
        torch.save(self.state_dict(), path_)