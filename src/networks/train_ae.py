import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
from ae import AutoEncoder
from utils import EarlyStopper
import sys
sys.path.append("../../")
from data.utils import MyDataset, Resize
from torcheval.metrics import MulticlassAccuracy, MulticlassConfusionMatrix, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, required=True, default = 100)
parser.add_argument('--lr', type=float, default=1e-3)

args = parser.parse_args()
epochs = args.epochs
lr = args.lr
dataset = MyDataset(root="../../data/Ventanas_augmented")
generator = torch.Generator().manual_seed(123)
train_dataset, validation_dataset, test_dataset =torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15], generator)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=8)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)

net = AutoEncoder()
early_stopper = EarlyStopper(patience=5, min_delta=0.01)
print(sum(p.numel() for p in net.parameters() if p.requires_grad))
net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
for epoch in tqdm(range(epochs)):
    for x, _ in train_loader:
        optimizer.zero_grad()
        pred = net(x.cuda())
        loss = net.loss(pred, x.cuda())
        loss.backward()
        optimizer.step()
    print(f"Training Loss: {net.epoch_loss/net.n}")
    
    for x, _ in validation_loader:
        pred = net(x.cuda())
        loss = net.val_loss(pred, x.cuda())
    
    print(f"Validation Loss: {net.val_epoch_loss/net.m}")
    if early_stopper.early_stop(net.val_epoch_loss):             
      print("Early stoppage at epoch:", epoch)
      break
    net.reset_losses()
net.finish(epochs)