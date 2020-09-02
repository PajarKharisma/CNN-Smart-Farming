import torch
import sys
import datetime
import numpy as np

class Param():
    #Hyperparameter
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    number_epochs = 100
    data_split = 0.9
    input_size = (331, 331)
    pretrained = False
    desc = 'FIRST TRY ' + str(datetime.datetime.now())