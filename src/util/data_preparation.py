import os
import cv2
import pandas as pd

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
