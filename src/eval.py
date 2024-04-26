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
from torcheval.metrics import MulticlassAccuracy, MulticlassConfusionMatrix


transform = transforms.Compose([Resize()])
dataset = MyDataset(root="../data/Ventanas", tform = transform)

train_dataset, validation_dataset, test_dataset =torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=8)

net = ModisTempNet()
net.load_state_dict(torch.load("networks/weights/Modis-Net_100.pth"))
net.eval()
print(sum(p.numel() for p in net.parameters() if p.requires_grad))

total_wins = 0
total = 0
metric = MulticlassAccuracy()
metric2 = MulticlassConfusionMatrix(20)
metric3 = MulticlassAccuracy(average=None, num_classes=20)
for x, y in tqdm(train_loader):
    pred = net(x)
    probs = net.softmax(pred)
    winners = probs.argmax(dim=1)
    target = y.argmax(dim=1)
    metric.update(winners, target)
    metric2.update(winners, target)
    metric3.update(winners, target)
print(metric.compute())
confusion_matrix = metric2.compute()
fig, ax = plt.subplots()
fig.set_figwidth(15)
fig.set_figheight(15)
# Using matshow here just because it sets the ticks up nicely. imshow is faster.
ax.matshow(confusion_matrix.numpy())

for (i, j), z in np.ndenumerate(confusion_matrix.numpy()):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
plt.show()
plt.savefig("cfm.png")
print(metric3.compute())
