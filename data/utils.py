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
from torchvision.transforms import v2
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
sys.path.append("../src/")
from networks.ModisTempNet import ModisTempNet

class MyDataset(torch.utils.data.Dataset):
    """Creates dataset that sampes: (landscape, class).
    Args:
        root (str): directory where data is being stored
        tform (Transform): tranformation to be applied at sampling time.
    """
    def __init__(self, root=".Ventanas_augmented", tform=None):
        super(MyDataset, self).__init__()
        self.root = root
        self.classification = pd.read_csv(f'{self.root}/database.csv')
        unique_classes = self.classification.clase.unique()
        self.translation = {}
        for i, j in zip(unique_classes, [k for k in range(len(unique_classes))]):
            self.translation[i] = j
        self.n_classes = len(self.translation)
        self.transform = tform
        imgs = os.listdir(f"{self.root}/")
        self.maxes = {}
        self.mins = {}
        for i in imgs:
            if i.split(".")[1] == "tif" and len(i.split(".")) == 2:
                src = rasterio.open(self.root + "/" + i)
                data = src.read()
                for band in range(data.shape[0]):
                    if band in self.maxes.keys():
                        if np.max(data[band]) > self.maxes[band]:
                            self.maxes[band] = np.max(data[band])
                        if np.min(data[band]) < self.mins[band]:
                            self.mins[band] = np.min(data[band])
                    else:
                        self.maxes[band] = np.max(data[band])
                        self.mins[band] = np.min(data[band])
        for band in range(data.shape[0]):
            self.mins[band] -= 1
        
        
    def __len__(self):
        return len(self.classification)
    
    def __getitem__(self, i):
        id_, class_ = self.classification.iloc[i]
        src = rasterio.open(f'{self.root}/ID{id_}.tif')
        data = src.read()
        n_class = self.translation[class_]
        y = torch.tensor([n_class], dtype=torch.int64)
        x = torch.from_numpy(data[:,:10,:10])
        n_bands = x.shape[0]
        for i in range(n_bands):
            x[i] = torch.where(torch.isnan(x[i]), self.mins[i], x[i])
        if torch.isnan(x).sum():
            print(x)
            print("Error")
        for band in range(n_bands):
            x[band] = (x[band] - self.mins[band]) / (self.maxes[band] - self.mins[band])
        if self.transform:
            x = self.transform(x)
        return x, y
        
    

class Resize(object):
    def __call__(self, x):
        x = transforms.functional.resize(x, size = 256, interpolation = transforms.InterpolationMode.NEAREST_EXACT)
        return x
    
