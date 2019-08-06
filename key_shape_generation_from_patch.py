# coding=utf-8

import kmeans
import os
import consts
import scipy.misc
import time
from skimage.feature import daisy

N = 1000000

def GenFeatures():
    PATCH_DIR = consts.SAVE_PATCH_PATH
    file_list = os.listdir(PATCH_DIR)
    t_start = time.clock()
    clip_num = int(min(len(file_list), N) / consts.COUNT_CLIP)
    for c in range(consts.COUNT_CLIP):
        data = []
        for i in range(c * clip_num, c * clip_num + clip_num):
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
        save_file_path = os.path.join(consts.SAVE_PATCH_CLUSTER_CENTER_PATH, consts.FEATURE_SAVE_NAME + str(c) + '.txt')
        with open(save_file_path, 'w+') as f:
            f.write(str(data))

def Cluster():
    import util_read_features
    feature_array, _ = util_read_features.ReadArray()
    print(feature_array.shape)
    model_kmeans = kmeans.KmeansModel()
    save_center_file_path = os.path.join(consts.SAVE_PATCH_CLUSTER_CENTER_PATH, consts.CLUSTER_CENTER_SAVE_NAME)
    cluster_res, centers = model_kmeans.cluster(feature_array, consts.K, save_file=save_center_file_path)
    print(centers)

if __name__ == '__main__':
    # GenFeatures()
    Cluster()