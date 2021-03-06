import torch

from src.config.path import *
from src.config.param import *

def save_checkpoint(**kwargs):
    data = kwargs
    checkpoint = {
        'title' : Param.title,
        'epoch': data['epoch'],
        'loss' : data['loss'],
        'class_names' : data['class_names'],
        'desc' : data['desc'],
        'solution' : data['solution'],
        'state_dict': data['model'].state_dict(),
        'optimizer': data['optimizer'].state_dict()
    }
    torch.save(checkpoint, data['save_dir'])

def load_checkpoint(load_dir):
    checkpoint = torch.load(load_dir, map_location=Param.device)

    return checkpoint