import os
import uuid

class Path():
    root_dir = os.getcwd()
    path = root_dir + '/dataset/'
    
    log_dir = root_dir + '/log/'
    save_model = root_dir + '/models/model-' + str(uuid.uuid4().hex) + '.pth'
    load_model = root_dir + '/models/model.pth'
    save_plot = root_dir+'/log/plot/'

    train_images = path + '/images/train/'
    validation_images = path + '/images/validation/'
    train_csv = path + '/train.csv'
    validation_csv = path + '/validation.csv'
