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
from networks.ConvNet import ConvNet
sys.path.append("../")
from data.utils import MyDataset, Resize
from torcheval.metrics import MulticlassAccuracy, MulticlassConfusionMatrix, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
import json

transform = transforms.Compose([Resize()])
dataset = MyDataset(root="../data/Ventanas", tform = transform)

generator = torch.Generator().manual_seed(123)
train_dataset, validation_dataset, test_dataset =torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15], generator)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=8)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)

net = ConvNet()
net.load_state_dict(torch.load("networks/weights/Conv-Net_100.pth"))
net.eval()
print(sum(p.numel() for p in net.parameters() if p.requires_grad))

net.cpu()
net.eval()
metric = MulticlassAccuracy()
metric2 = MulticlassConfusionMatrix(20)
metric3 = MulticlassAccuracy(average=None, num_classes=20)
metric4 = MulticlassF1Score(average="macro", num_classes=20)
metric5 = MulticlassPrecision(average="macro", num_classes=20)
metric6 = MulticlassRecall(average="macro", num_classes=20)
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
with open(f'plots/results_val_{net.name}_{100}.json', 'w') as f:
    json.dump(results, f)

confusion_matrix = metric2.compute()
fig, ax = plt.subplots()
fig.set_figwidth(15)
fig.set_figheight(15)
# Using matshow here just because it sets the ticks up nicely. imshow is faster.
ax.matshow(confusion_matrix.numpy())

for (i, j), z in np.ndenumerate(confusion_matrix.numpy()):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
plt.savefig(f"plots/cfm_val_{net.name}_{100}.png")
