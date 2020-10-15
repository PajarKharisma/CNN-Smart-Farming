import sys
import os
import time
from datetime import datetime
root_dir = os.getcwd()
sys.path.append(root_dir)

import copy
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
import src.util.read_desc as read_desc

import src.nnarch.simple_cnn as simple_cnn
import src.nnarch.nasnet as nasnet
import src.nnarch.nasnet_mobile as nasnet_mobile
import src.nnarch.bstcnn as bstcnn

from src.config.path import *
from src.config.param import *

def create_dataset():
    data_preparation.create_csv(dir_csv=Path.train_csv, dir_images=Path.train_images)
    # data_preparation.data_augmentation(dir_images=Path.train_images)

def dataset_load(split_data=False, validation_data_exist=False):
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
            validation_dataloader = DataLoader(val_set, batch_size=Param.batch_size, shuffle=True)

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

def training(model, loss_function, dataset, optimizer, loss, epoch_number=0, verbose=False):
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

    for data in train_loader:
        images, labels = data
        print(images.size())
    
    return
    for epoch in range(Param.number_epochs):
        train_loss = 0
        train_acc = 0
        for i, data in enumerate(train_loader):
            print(i)
            model.train()

            images, labels = data

            images = images.to(Param.device)
            labels = labels.to(Param.device)
            print(images.size())
            # print(labels.size())
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss = train_loss + ((loss.item() - train_loss) / (i + 1))
            acc = metrics.get_acc(model, [data])
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
        desc=read_desc.get_desc(Path.desc_disease),
        solution=read_desc.get_solution(Path.solution_disease),
        epoch=Param.number_epochs + epoch_number,
        loss=best_loss,
        class_names=dataset_loader.get_class_names(Path.train_csv)
    )

def print_model():
    model = nasnet.NASNetALarge(num_classes=4)
    model = model.to(Param.device)
    summary(model, (3, 331, 331))

def main():
    # model = nasnet.NASNetALarge(num_classes=4)
    # model = simple_cnn.ConvNet(num_classes=4)
    model = bstcnn.BstCnn(num_classes=4)
    # model = nasnet_mobile.NASNetAMobile(num_classes=4)
    model = model.to(Param.device)

    optimizer = optim.Adam(model.parameters())
    epoch = 0
    loss = sys.float_info.max

    if(Param.pretrained == True):
        checkpoint  = ckp.load_checkpoint(load_dir=Path.load_model)
        
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    criterion = nn.CrossEntropyLoss()
    sys.stdout.write('# READING DATASET\n')

    dataset = dataset_load(split_data=True,validation_data_exist=False)

    sys.stdout.write('# FINISHED READING DATASET AND START TRAINING\n\n')
    sys.stdout.flush()

    training(
        model=model,
        loss_function=criterion,
        dataset=dataset,
        optimizer=optimizer,
        loss=loss,
        epoch_number=epoch,
        verbose=False
    )

if __name__ == "__main__":
    start_time = time.time()
    sys.stdout.write('Process using '+str(Param.device)+'\n')
    sys.stdout.write(Param.title+'\n\n')

    # create_dataset()
    # print_model()
    main()

    elapsed_time = time.time() - start_time
    sys.stdout.write(time.strftime("Finish in %H:%M:%S\n", time.gmtime(elapsed_time)))