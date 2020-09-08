import os

def get_desc(dir_desc):
    results = []

    files = os.listdir(dir_desc)
    for file in files:
        full_path = os.path.join(dir_desc,file)
        with open(full_path) as reader:
            results.append(reader.read())

    return results