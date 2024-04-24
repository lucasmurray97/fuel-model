import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
import pickle
from matplotlib import pyplot as plt
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import DataLoader
import fiona
import rasterio
import rasterio.mask
import pathlib
import pandas as pd
import os
import rioxarray
import sys
from tqdm import tqdm
from networks.ModisTempNet import ModisTempNet
sys.path.append("../")
from data.utils import MyDataset, Resize


transform = transforms.Compose([Resize()])
dataset = MyDataset(root="../data/Ventanas", tform = transform)

train_dataset, validation_dataset, test_dataset =torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=8)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)

net = ModisTempNet()
net.cuda()

epochs = 20
optimizer = torch.optim.Adam(net.parameters())
for epoch in tqdm(range(epochs)):
    for x, y in train_loader:
        optimizer.zero_grad()
        pred = net(x.cuda())
        loss = net.train_loss(pred, y.cuda())
        loss.backward()
        optimizer.step()
    print(f"Training Loss: {net.epoch_loss/net.n}")
    for x, y in validation_loader:
        pred = net(x.cuda())
        loss = net.val_loss(pred, y.cuda())
    print(f"Validation Loss: {net.val_epoch_loss/net.m}")
    net.reset_losses()
net.finish(epochs)

