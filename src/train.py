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
from networks.ResNet import ResNet_18
from networks.ResNet34 import ResNet_34
from networks.ConvNet import ConvNet
from networks.ResNet_pre import ResNet_Pre_18
from networks.utils import EarlyStopper
sys.path.append("../")
from data.utils import MyDataset, Resize
from torcheval.metrics import MulticlassAccuracy, MulticlassConfusionMatrix, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, required=True, default = 100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)

args = parser.parse_args()
epochs = args.epochs
lr = args.lr
wd = args.weight_decay
transform = transforms.Compose([])
dataset = MyDataset(root="../data/Ventanas_augmented", tform = transform)
print(len(dataset))
generator = torch.Generator().manual_seed(123)
train_dataset, validation_dataset, test_dataset =torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15], generator)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, drop_last=True)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)

early_stopper = EarlyStopper(patience=5, min_delta=0.01)
net = ResNet_Pre_18()
print(sum(p.numel() for p in net.parameters() if p.requires_grad))
net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
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
    if early_stopper.early_stop(net.val_epoch_loss):             
      print("Early stoppage at epoch:", epoch)
      break
    net.reset_losses()
net.finish(epochs)

net.cpu()
net.eval()
metric = MulticlassAccuracy()
metric2 = MulticlassConfusionMatrix(20)
metric3 = MulticlassAccuracy(average=None, num_classes=20)
metric4 = MulticlassF1Score(average="macro", num_classes=20)
metric5 = MulticlassPrecision(average="macro", num_classes=20)
metric6 = MulticlassRecall(average="macro", num_classes=20)
for x, y in tqdm(train_loader):
    pred = net(x)
    probs = net.softmax(pred)
    winners = probs.argmax(dim=1)
    target = y.squeeze(1)
    metric.update(winners, target)
    metric2.update(winners, target)
    metric3.update(winners, target)
    metric4.update(winners, target)
    metric5.update(winners, target)
    metric6.update(winners, target)
results = {}
results["accuracy"] = metric.compute().item()
results["accuracy_per_class"] = metric3.compute().tolist()
results["f1"] = metric4.compute().item()
results["precision"] = metric5.compute().item()
results["recall"] = metric6.compute().item()
with open(f'plots/results_{net.name}_{epochs}.json', 'w') as f:
    json.dump(results, f)

confusion_matrix = metric2.compute()
fig, ax = plt.subplots()
fig.set_figwidth(15)
fig.set_figheight(15)
# Using matshow here just because it sets the ticks up nicely. imshow is faster.
ax.matshow(confusion_matrix.numpy())

for (i, j), z in np.ndenumerate(confusion_matrix.numpy()):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
plt.savefig(f"plots/cfm_{net.name}_{epochs}.png")

metric.reset()
metric2.reset()
metric3.reset()
metric4.reset()
metric5.reset()
metric6.reset()
for x, y in tqdm(validation_loader):
    pred = net(x)
    probs = net.softmax(pred)
    winners = probs.argmax(dim=1)
    target = y.squeeze(1)
    metric.update(winners, target)
    metric2.update(winners, target)
    metric3.update(winners, target)
    metric4.update(winners, target)
    metric5.update(winners, target)
    metric6.update(winners, target)
results = {}
results["accuracy"] = metric.compute().item()
results["accuracy_per_class"] = metric3.compute().tolist()
results["f1"] = metric4.compute().item()
results["precision"] = metric5.compute().item()
results["recall"] = metric6.compute().item()
with open(f'plots/results_val_{net.name}_{epochs}.json', 'w') as f:
    json.dump(results, f)

confusion_matrix = metric2.compute()
fig, ax = plt.subplots()
fig.set_figwidth(15)
fig.set_figheight(15)
# Using matshow here just because it sets the ticks up nicely. imshow is faster.
ax.matshow(confusion_matrix.numpy())

for (i, j), z in np.ndenumerate(confusion_matrix.numpy()):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
plt.savefig(f"plots/cfm_val_{net.name}_{epochs}.png")

