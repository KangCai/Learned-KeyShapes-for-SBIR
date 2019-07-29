# coding=utf-8

import kmeans
import numpy as np
import re
import os
import scipy.misc
from skimage.feature import daisy
import time
import matplotlib.pyplot as plt

import stroke_points_detection

DATASET_PNG_PATH = 'D:/game/G78/AI/data/sketch_data/png/'
SAVE_PATCH_PATH = 'D:/game/G78/AI/data/sketch_data/png_patch/'
SAVE_PATCH_CLUSTER_CENTER_PATH = 'D:/game/G78/AI/data/sketch_data/png_patch_cluster_center/'
PATCH_SIZE = 31
W, H = 400, 400

def KeyShapeGeneration():
    # Generate a million of 31 x 31 sketch patches; Generate a daisy descriptor with respect to each center of patch.
    data = _PatchDaisyGeneration()
    # Cluster all descriptors by Kmeans algorithm
    _Cluster(data)

def _PatchDaisyGeneration():
    data = []
    with open(os.path.join(DATASET_PNG_PATH, 'filelist.txt')) as f:
        total_count, total_patch, total_time = 0, 0, 0
        t_start = time.clock()
        while True:
            target_sketch_image = f.readline().strip('\n')
            if not target_sketch_image:
                break
            image_abs_path = os.path.join(DATASET_PNG_PATH, target_sketch_image)
            try:
                image_array = scipy.misc.imread(image_abs_path)
                # plt.imshow(image_array, cmap ='gray')
                # plt.show()
                image_array = scipy.misc.imresize(image_array, size=(W, H))
                # plt.imshow(image_array, cmap ='gray')
                # plt.show()
                image_row, image_col = image_array.shape
                # Extract patch from image
                N_ = 50
                stroke_points = stroke_points_detection.GetStrokePointsHarris(image_array, N_)
                # print('Count of detected stroke points is %r/%r' % (len(stroke_points), N_))
                # print(stroke_points)
                (filename, extension) = os.path.splitext(target_sketch_image)
                for p_idx, stroke_point in enumerate(stroke_points):
                    x, y, _ = stroke_point
                    x_l, x_r, y_l, y_r = x - int(PATCH_SIZE / 2), x + int(PATCH_SIZE / 2), y - int(
                        PATCH_SIZE / 2), y + int(PATCH_SIZE / 2)
                    if x_l < 0 or x_r >= image_row or y_l < 0 or y_r >= image_col:
                        continue
                    patch = image_array[x_l:x_r + 1, y_l:y_r + 1]
                    daisy_descriptor = daisy(patch, rings=2)
                    # Save image patch into file
                    patch_file_path = os.path.join(SAVE_PATCH_PATH, filename.replace('/', '_')+'_%d%s'%(p_idx, extension))
                    scipy.misc.imsave(patch_file_path, patch)
                    data.append((daisy_descriptor[0][0], patch_file_path))
                    total_patch += 1
                total_count += 1
                if total_count % 100 == 0:
                    print('%r files processed, %r patches generated, %ds cost' % (total_count, total_patch, int(time.clock() - t_start)))
            except Exception as e:
                print(e, image_abs_path)
    return data

def _Cluster(data):
    feature_list = [d[0] for d in data]
    model_kmeans = kmeans.KmeansModel()
    K = 150
    cluster_res, centers = model_kmeans.cluster(np.array(feature_list), K)
    centers_list = [list(i) for i in centers]
    with open(os.path.join(SAVE_PATCH_CLUSTER_CENTER_PATH, 'cluster_centers.txt'), 'w+') as f:
        f.write(str(centers_list))
    from collections import defaultdict
    cluster_dict = defaultdict(list)
    for i in range(len(cluster_res)):
        label, point = cluster_res[i]
        cluster_dict[label].append((i, point))
    cluster_center_patch = np.zeros((K, PATCH_SIZE, PATCH_SIZE))
    for label, point_list in cluster_dict.items():
        print(label, len(point_list))
        for idx, point in point_list:
            patch_file_path = data[idx][1]
            image_array = scipy.misc.imread(patch_file_path)
            cluster_center_patch[label, :, :] += image_array
        cluster_center_patch[label, :, :] /= len(point_list)
        cluster_center_patch[label, :, :] = 255.0 - cluster_center_patch[label, :, :]
        plt.matshow(cluster_center_patch[label, :, :], cmap='jet')
        plt.axis('off')
        plt.savefig(os.path.join(SAVE_PATCH_CLUSTER_CENTER_PATH, '%r.png'%label))
    # _ShowClusterInfo(cluster_res, low_dim_data_mat)

def _ShowClusterInfo(cluster_res, feature_list):
    import principal_component_analysis
    low_dim_data_mat, _ = principal_component_analysis.pca(np.array(feature_list), top_n_feat=2)
    c_list = ['r', 'g', 'b', 'm', 'c']
    for i in range(low_dim_data_mat.shape[0]):
        print(low_dim_data_mat[i, :])
        x, y = low_dim_data_mat[i, :]
        plt.scatter(x, y, c=c_list[cluster_res[i][0]])

if __name__ == '__main__':
    KeyShapeGeneration()