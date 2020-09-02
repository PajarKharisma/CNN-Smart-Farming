import os
import pandas as pd
import uuid
import cv2

from PIL import Image

def create_csv(dir_csv, dir_images):
    class_names = os.listdir(dir_images)
    dataset = []

    for class_name in class_names:
        full_path = os.path.join(dir_images, class_name).replace("\\","/")
        files = os.listdir(full_path)

        for file in files:
            data = {}
            data['img'] = os.path.join(class_name, file).replace("\\","/")
            data['disease_name'] = class_name
            data['disease_index'] = class_names.index(class_name)
            dataset.append(data)

    df = pd.DataFrame(dataset)
    df.to_csv(dir_csv, index=False)

def data_augmentation(dir_images):
    class_names = os.listdir(dir_images)

    for class_name in class_names:
        full_path = os.path.join(dir_images, class_name).replace("\\","/")
        files = os.listdir(full_path)

        for file in files:
            img = cv2.imread('{}/{}'.format(full_path, file))

            degree = 90
            for i in range(3):
                alias = file.split('.')[0]
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite('{}/{}_rotate_{}.jpg'.format(full_path, alias, degree), img)
                degree += 90


def resize_img(dir_images):
    class_names = os.listdir(dir_images)

    files_count = {
        'blight':96,
        'blast':80,
        'brownspot':100
    }

    for class_name in class_names:
        full_path = os.path.join(dir_images, class_name).replace("\\","/")
        files = os.listdir(full_path)

        order = files_count[class_name]
        for file in files:
            order += 1
            file_name = os.path.join(full_path, file).replace("\\","/")
            img = Image.open(file_name)
            img = img.resize((300,300))
            alias = '{}_orig_{}'.format(class_name, str(order).zfill(3))
            img.save('{}/{}.jpg'.format(full_path, alias))

def rename_img(dir_images):
    class_names = os.listdir(dir_images)

    for class_name in class_names:
        full_path = os.path.join(dir_images, class_name).replace("\\","/")
        files = os.listdir(full_path)

        for file in files:
            file_name = os.path.join(full_path, file).replace("\\","/")
            img = cv2.imread(file_name)
            os.remove(file_name)
            alias = file.split('.')[0]

            cv2.imwrite('{}/{}.jpg'.format(full_path, alias), img)
