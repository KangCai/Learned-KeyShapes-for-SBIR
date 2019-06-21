# coding=utf-8

import kmeans
import numpy as np
import re
import os
import scipy.misc
from skimage.feature import daisy

DATASET_SVG_PATH = 'D:/game/G78/AI/data/sketch_data/svg/'
DATASET_PNG_PATH = 'D:/game/G78/AI/data/sketch_data/png/'
PATCH_SIZE = 31

def KeyShapeGeneration():
    # Generate a million of 31 x 31 sketch patches; Generate a daisy descriptor with respect to each center of patch.
    _PatchDaisyGeneration()
    # Cluster all descriptors by Kmeans algorithm
    _Cluster()

def _PatchDaisyGeneration():
    with open(os.path.join(DATASET_SVG_PATH, 'filelist.txt')) as f:
        total_count = 0
        while True:
            target_sketch_image = f.readline().strip('\n')
            if not target_sketch_image:
                break
            with open(os.path.join(DATASET_SVG_PATH, target_sketch_image)) as ft:
                image_array = scipy.misc.imread(os.path.join(DATASET_PNG_PATH, os.path.splitext(target_sketch_image)[0] + '.png'))
                image_row, image_col = image_array.shape
                svg_content = ft.read()
                result = re.findall(' d=\"(.*) \"/>', svg_content)
                x_last, y_last = None, None
                for path in result:
                    l = re.findall(r'[0-9.]+', path)
                    for i in range(int(len(l) / 2)):
                        x1, y1 = int(float(l[i * 2])), int(float(l[i * 2 + 1]))
                        if (x_last is not None and y_last is not None) and ((x_last - x1) ** 2 + (y_last - y1) ** 2) ** 0.5 < 31:
                            continue
                        x_l, x_r, y_l, y_r = x1 - int(PATCH_SIZE / 2), x1 + int(PATCH_SIZE / 2), y1 - int(PATCH_SIZE / 2), y1 + int(PATCH_SIZE / 2)
                        if x_l < 0 or x_r >= image_row or y_l < 0 or y_r >= image_col:
                            continue
                        daisy_descriptor = daisy(image_array[x_l:x_r+1, y_l:y_r+1], rings=2)
                        print(daisy_descriptor)
                        total_count += 1
                        x_last, y_last = x1, y1

def _Cluster():
    model_kmeans = kmeans.KmeansModel()
    for label, point in model_kmeans.cluster(np.array([[0, 1], [1, 1], [2, 2], [4, 5]]), 2):
        print(label, point)

if __name__ == '__main__':
    KeyShapeGeneration()