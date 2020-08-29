import numpy as np
import matplotlib.pyplot as plt

def show_plot(**kwargs):
    data = kwargs

    plt.plot(data['epoch'], data['train_data'])
    plt.plot(data['epoch'], data['val_data'])
    plt.title(data['title'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'val'], loc=data['legend_loc'])
        
    if data['should_show']:
        plt.show()
    if data['should_save']:
        plt.savefig(data['path'])
    plt.close()