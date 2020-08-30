import pandas as pd
import numpy as np

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, csv_path, images_path, transform=None, count=-1):
        self.images_path = images_path
        self.csv_path = csv_path
        self.transform = transform
        self.df = pd.read_csv(self.csv_path)
        self.count = count
    
    def __getitem__(self, index):
        img = Image.open(self.images_path + self.df['img'][index])
        label = int(self.df['disease_index'][index])

        if transforms is not None:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        if self.count == -1:
            return len(self.df)
        else:
            return self.count

def get_class_names(csv_path):
    df = pd.read_csv(csv_path)
    return sorted(set(df['disease_names']))