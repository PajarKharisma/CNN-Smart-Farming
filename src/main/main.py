import sys
import os
import time
from datetime import datetime
root_dir = os.getcwd()
sys.path.append(root_dir)

import pandas as pd

import torch
import torch.nn as nn
import torchvision.utils
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import summary

import src.util.data_preparation as data_preparation
import src.util.dataset_loader as dataset_loader
import src.util.metrics as metrics
import src.util.visual as vis
import src.util.checkpoint as ckp

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

            train_dataloader = DataLoader(train_set, batch_size=Param.batch_size, shuffle=True)
            validation_dataloader = DataLoader(val_set, batch_size=Param.batch_size * 2, shuffle=True)

            return train_dataloader, validation_dataloader
        else:
            return DataLoader(dataset, batch_size=Param.batch_size, shuffle=True)

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

        train_dataloader = DataLoader(train_dataset, batch_size=Param.batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=Param.batch_size, shuffle=True)

        return train_dataloader, validation_dataloader

def training(model, loss_function, dataset, optimizer, loss, verbose=False, epoch_number=0):
    criterion = loss_function
    train_loader, validation_loader = dataset
    optimizer = optimizer

    history_loss = {
        'epoch' : [],
        'train' : [],
        'val' : []
    }

    history_acc = {
        'epoch' : [],
        'train' : [],
        'val' : []
    }

    best_loss = loss
    val_model = None
    best_model = None
    total_step = len(train_loader)

    if Param.pretrained == True:
        best_model = copy.deepcopy(model)

    for epoch in range(Param.number_epochs):
        train_loss = 0
        train_acc = 0
        for i, (images, labels) in enumerate(train_loader):
            model.train()

            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss = train_loss + ((loss.item() - train_loss) / (i + 1))
            acc = metrics.get_acc(model, train_loader)
            train_acc = train_acc + ((acc - train_acc) / (i + 1))

            if verbose and (i+1) % 100 == 0:
                sys.stdout.write('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}\n'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
        if train_loss < best_loss:
            best_loss = train_loss
            best_model = copy.deepcopy(model)

        validation_loss = metrics.get_loss(model, validation_loader, criterion)
        validation_acc = metrics.get_acc(model, validation_loader)

        output_str = ''
        output_str += 'EPOCH {} SUMMARY : \n'.format(epoch + 1)
        output_str += '-'*40 + '\n'
        output_str += 'Train loss : {}\n'.format(train_loss)
        output_str += 'Validation loss : {}\n'.format(validation_loss)
        output_str += 'Train Acc : {}\n'.format(train_acc)
        output_str += 'Validation Acc : {}\n'.format(validation_acc)
        output_str += '='*40 + '\n'

        sys.stdout.write(output_str)
        sys.stdout.flush()

        history_loss['epoch'].append(epoch+1)
        history_loss['train'].append(train_loss)
        history_loss['val'].append(validation_loss)

        history_acc['epoch'].append(epoch+1)
        history_acc['train'].append(train_acc)
        history_acc['val'].append(validation_acc)

        vis.show_plot(
            epoch=history_loss['epoch'],
            train_data=history_loss['train'],
            val_data=history_loss['val'],
            title='Loss Train dan Validasi',
            xlabel='Epoch',
            ylabel='Loss',
            legend=['Train', 'Val'],
            path=Path.save_plot+'Model Loss.png',
            should_show=False,
            should_save=True,
            legend_loc='upper right'
        )

        vis.show_plot(
            epoch=history_loss['epoch'],
            train_data=history_acc['train'],
            val_data=history_acc['val'],
            title='Akurasi Train dan Validasi',
            xlabel='Epoch',
            ylabel='Akurasi',
            legend=['Train', 'Val'],
            path=Path.save_plot+'Model Akurasi.png',
            should_show=False,
            should_save=True,
            legend_loc='lower right'
        )

        ckp.save_checkpoint(
            save_dir=Path.save_model,
            model=best_model,
            optimizer=optimizer,
            epoch=Param.train_number_epochs + epoch_number,
            loss=best_loss,
            class_names=dataset_loader.get_class_names(Path.train_csv)
        )

if __name__ == "__main__":
    start_time = time.time()
    sys.stdout.write('Process using '+str(Param.device)+'\n')
    sys.stdout.write(Param.desc+'\n\n')

    train_dataset, validation_dataset = dataset_load(validation_data_exist=True)

    elapsed_time = time.time() - start_time
    sys.stdout.write(time.strftime("Finish in %H:%M:%S\n", time.gmtime(elapsed_time)))