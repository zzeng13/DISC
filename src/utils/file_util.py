import os
import json
import numpy as np


def load_npy_file(path):
    return np.load(path)


def write_npy_file(path, np_array):
    np.save(path, np_array)
    return


def load_txt_file(path):
    with open(path, 'r') as f:
        return f.read().splitlines()


def write_txt_file(path, content):
    with open(path, 'w') as f:
        for line in content:
            f.write(line)
            f.write('\n')
        f.close()
    return


def load_json_file(path):
    with open(path, 'r') as f:
        return json.load(f)


def write_json_file(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)
    return


def check_make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Created directory: {}'.format(path))
    else:
        print('Directory {} already exists!'.format(path))

