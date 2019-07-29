# coding=utf-8

import kmeans
import os
import key_shape_generation_from_original
import scipy.misc

def Work():
    PATCH_DIR = key_shape_generation_from_original.SAVE_PATCH_PATH
    file_list = os.listdir(PATCH_DIR)
    for i in range(len(file_list)):
        (filename, extension) = os.path.splitext(file_list[i])
        if extension != '.png':
            continue
        full_path = os.path.join(PATCH_DIR, file_list[i])
        print(full_path)
        image_array = scipy.misc.imread(full_path)

if __name__ == '__main__':
    Work()