# coding=utf-8

import kmeans
import os
import key_shape_generation_from_original
import scipy.misc
import time
import numpy as np
from skimage.feature import daisy
from sklearn.cluster import KMeans

def GenFeatures():
    PATCH_DIR = key_shape_generation_from_original.SAVE_PATCH_PATH
    file_list = os.listdir(PATCH_DIR)
    t_start = time.clock()
    data = []
    for i in range(100000):
        try:
            (filename, extension) = os.path.splitext(file_list[i])
            if extension != '.png':
                continue
            full_path = os.path.join(PATCH_DIR, file_list[i])
            image_array = scipy.misc.imread(full_path)
            daisy_descriptor = list(daisy(image_array, rings=2)[0][0])
            data.append((daisy_descriptor, file_list[i]))
        except Exception as e:
            print(i, file_list[i], e)
        if i % 1000 == 0:
            print('%r files processed, %r valid features generated, %ds cost' % (i, len(data), int(time.clock() - t_start)))
    save_file_path = os.path.join(key_shape_generation_from_original.SAVE_PATCH_CLUSTER_CENTER_PATH, 'features_total.txt')
    with open(save_file_path, 'w+') as f:
        f.write(str(data))
    # with open(save_file_path) as f:
    #     a = f.read()
    #     print(len(eval(a)))

def Cluster():
    save_feature_file_path = os.path.join(key_shape_generation_from_original.SAVE_PATCH_CLUSTER_CENTER_PATH, 'features.txt')
    with open(save_feature_file_path) as f:
        a = f.read()
        feature_list = np.array([i[0] for i in eval(a)])
        print(feature_list.shape)
        model_kmeans = kmeans.KmeansModel()
        K = 150
        save_center_file_path = os.path.join(key_shape_generation_from_original.SAVE_PATCH_CLUSTER_CENTER_PATH, 'cluster_centers.txt')
        cluster_res, centers = model_kmeans.cluster(feature_list, K, save_file=save_center_file_path)
        print(centers)

if __name__ == '__main__':
    Cluster()