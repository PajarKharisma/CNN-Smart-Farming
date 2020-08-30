import torch
import torch.nn.functional as F
from torch.autograd import Variable

import copy
import numpy as np

from src.config.param import *

def get_acc(model, dataset):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataset:
            images = images.to(Param.device)
            labels = labels.to(Param.device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return (correct / total) * 100

def get_loss(model, dataset, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(Param.device)
            labels = labels.to(Param.device)
            
            outputs = model(images)
            loss = criterion(outputs, labels).item()
            
            total_loss = total_loss + ((loss - total_loss) / (i + 1))

    return total_loss