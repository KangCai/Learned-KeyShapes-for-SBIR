# coding=utf-8

import numpy as np
import consts
import os
import util_read_features

def Assignment():
    # 读取 centers
    save_center_file_path = os.path.join(consts.SAVE_PATCH_CLUSTER_CENTER_PATH, consts.CLUSTER_CENTER_SAVE_NAME)
    with open(save_center_file_path) as f:
        center_list = eval(f.read())
        centers = np.array(center_list)
        print(centers)
        # 读取 x
        x, _ = util_read_features.ReadArray()
        row, col = x.shape
        assignments, assignments_reversed = np.zeros(row), [[] for _ in range(consts.K)]
        for i in range(row):
            dists = np.linalg.norm(x[i] - centers, axis=1)
            assignments[i] = int(np.argmin(dists))
            assignments_reversed[int(np.argmin(dists))].append(i)
        for i in range(consts.K):
            print(i, len(assignments_reversed[i]))
    return assignments, assignments_reversed

def DrawCluster():
    import matplotlib.pyplot as plt
    import scipy.misc
    # 读取 centers
    save_center_file_path = os.path.join(consts.SAVE_PATCH_CLUSTER_CENTER_PATH, consts.CLUSTER_CENTER_SAVE_NAME)
    with open(save_center_file_path) as f:
        center_list = eval(f.read())
        centers = np.array(center_list)
        print(centers)
    # 读取
    x, file_list = util_read_features.ReadArray()
    row, col = x.shape
    cluster_center_patch = np.zeros((consts.K, consts.PATCH_SIZE, consts.PATCH_SIZE))
    assignments_reversed = [[] for _ in range(consts.K)]
    for i in range(row):
        dists = np.linalg.norm(x[i] - centers, axis=1)
        label = int(np.argmin(dists))
        assignments_reversed[label].append(i)
        file_name = file_list[i]
        patch_file_path = os.path.join(consts.SAVE_PATCH_PATH, file_name)
        image_array = scipy.misc.imread(patch_file_path)
        cluster_center_patch[label, :, :] += image_array
        if i % 1000 == 0:
            print('Patch image average done %r/%r' % (i, row))
    plt.figure(figsize=(16, 3.4))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    for label in range(consts.K):
        ax = plt.subplot(6, 25, label + 1)
        cluster_center_patch[label, :, :] /= len(assignments_reversed[label])
        cluster_center_patch[label, :, :] = 255.0 - cluster_center_patch[label, :, :]
        ax.matshow(cluster_center_patch[label, :, :], cmap='jet')
        ax.axis('off')
    plt.show()

if __name__ == '__main__':
    # Assignment()
    DrawCluster()