import torch
import sys
import datetime
import numpy as np

class Param():
    #Hyperparameter
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    number_epochs = 100
    data_split = 0.9
    input_size = (300, 300)
    pretrained = False
    title = 'BST CNN ' + str(datetime.datetime.now())