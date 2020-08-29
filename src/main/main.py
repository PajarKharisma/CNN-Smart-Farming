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

def dataset_load(split_data = False, validation_data_exist=False):
    transform = transforms.Compose([transforms.Resize(Param.input_size), transforms.ToTensor()])

    if validation_data_exist == False:
        dataset = dataset_loader.ImageDataset(
            csv_path=Path.train_csv,
            images_path=Path.train_images,
            transform=transform
        )

        if split_data:
            train_length = int(len(dataset) * Param.data_split)
            val_length = len(dataset) - train_length

            train_set, val_set = torch.utils.data.random_split(dataset, [train_length, val_length])

            train_dataloader = DataLoader(train_set, batch_size=Param.train_batch_size, shuffle=True)
            validation_dataloader = DataLoader(val_set, batch_size=Param.train_batch_size * 2, shuffle=True)

            return train_dataloader, validation_dataloader
        else:
            return DataLoader(dataset, batch_size=Param.train_batch_size, shuffle=True)

    else:
        train_dataset = dataset_loader.ImageDataset(
            csv_path=Path.train_csv,
            images_path=Path.train_images,
            transform=transform
        )

        validation_dataset = dataset_loader.ImageDataset(
            csv_path=Path.validation_csv,
            images_path=Path.validation_images,
            transform=transform
        )

        train_dataloader = DataLoader(train_dataset, batch_size=Param.train_batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=Param.train_batch_size, shuffle=True)

        return train_dataloader, validation_dataloader

def training(model, loss_function, dataset, optimizer, loss, epoch_number=0):
    pass

if __name__ == "__main__":
    start_time = time.time()
    sys.stdout.write('Process using '+str(Param.device)+'\n')
    sys.stdout.write(Param.desc+'\n\n')

    train_dataset, validation_dataset = dataset_load(validation_data_exist=True)
    for img, label in validation_dataset:
        print(img.size())
        print(label)
        print('-'*30)

    elapsed_time = time.time() - start_time
    sys.stdout.write(time.strftime("Finish in %H:%M:%S\n", time.gmtime(elapsed_time)))