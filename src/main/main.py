import sys
import os
import time
from datetime import datetime
root_dir = os.getcwd()
sys.path.append(root_dir)

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import src.util.data_preparation as data_preparation
import src.util.dataset_loader as dataset_loader

from src.config.path import *
from src.config.param import *

def create_dataset():
    data_preparation.create_csv(dir_csv=Path.validation_csv, dir_images=Path.validation_images)

def dataset_load(split_data = True):
    transform = transforms.Compose([transforms.Resize(Param.input_size), transforms.ToTensor()])
    dataset = dataset_loader.ImageDataset(
        csv_path=Path.train_csv,
        images_path=Path.train_images,
        transform=transform
    )

    return DataLoader(dataset, batch_size=Param.train_batch_size, shuffle=True)

if __name__ == "__main__":
    start_time = time.time()
    sys.stdout.write('Process using '+str(Param.device)+'\n')
    sys.stdout.write(Param.desc+'\n\n')

    dataset = dataset_load()
    for img, label in dataset:
        print(img.size())
        print(label)
        print('-'*30)

    elapsed_time = time.time() - start_time
    sys.stdout.write(time.strftime("Finish in %H:%M:%S\n", time.gmtime(elapsed_time)))