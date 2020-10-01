import os
import uuid

class Path():
    root_dir = os.getcwd()
    path = root_dir + '/dataset/hama/'
    
    log_dir = root_dir + '/log/'
    save_model = root_dir + '/models/model-' + str(uuid.uuid4().hex) + '.pth'
    load_model = root_dir + '/models/model.pth'
    save_plot = root_dir+'/log/plot/'

    desc_disease = path + '/desc'
    solution_disease = path + '/solution'
    train_images = path + '/images/'
    validation_images = path + '/images/validation/'
    train_csv = path + '/train.csv'
    validation_csv = path + '/validation.csv'