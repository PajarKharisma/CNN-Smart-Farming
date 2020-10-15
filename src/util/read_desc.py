import os

def get_desc(dir_desc):
    results = []

    files = os.listdir(dir_desc)
    for file in files:
        full_path = os.path.join(dir_desc,file)
        with open(full_path) as reader:
            results.append(reader.read())

    return results

def get_solution(dir_solution):
    results = []

    files = os.listdir(dir_solution)
    for file in files:
        full_path = os.path.join(dir_solution,file)
        with open(full_path) as reader:
            result = reader.read().replace('\n','').split('{}')
            result = [i for i in result if i]
            results.append(result)

    return results