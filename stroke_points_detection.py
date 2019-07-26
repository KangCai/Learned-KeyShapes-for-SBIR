# coding=utf-8

import numpy as np

W = None

def GetStrokePointsHarris(image_data, N, grid_len=10):
    """

    :param image_data:
    :param N:
    :param grid_len:
    :return:
    """
    global W
    all_points = np.where(image_data < 100)
    count_all_points = len(np.where(image_data < 100)[0])
    sigma, window_size = 1, 31
    u = v = int((window_size - 1) / 2)
    if W is None:
        W = np.zeros((2*u+1, 2*v+1))
        for p in range(-u, u + 1):
            for q in range(-v, v + 1):
                W[p, q] = np.e ** (-(p ** 2 + q ** 2) / sigma ** 2)
    E = []
    for i in range(count_all_points):
        x, y = all_points[0][i], all_points[1][i]
        E_diff_matrix = (image_data[x - u:x + u + 1, y - v:y + v + 1].astype(int) - image_data[x, y])**2
        E_Harris_matrix = E_diff_matrix * W
        E_Harris = np.sum(E_Harris_matrix)
        E.append((x, y, E_Harris))
    E = sorted(E, key=lambda a: a[2], reverse=True)
    # Filter dense points.
    row, col = image_data.shape
    count_grid = row / grid_len
    E_filtered = []
    recorder = set()
    for info in E:
        x, y, _ = info
        key = _GetKey(x, y, grid_len, count_grid)
        if key in recorder:
            continue
        _AddKeyAround(recorder, x, y, grid_len, count_grid)
        E_filtered.append(info)
        if len(E_filtered) >= N:
            break
    return E_filtered

def _GetKey(x, y, grid_len, count_grid):
    return round(x / grid_len) * count_grid + round(y / grid_len)

def _AddKeyAround(recorder, x, y, grid_len, count_grid):
    key_x, key_y = round(x / grid_len), round(y / grid_len)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            recorder.add((key_x + dx) * count_grid + key_y + dy)

if __name__ == '__main__':
    import scipy.misc
    import os
    import matplotlib.pyplot as plt
    DATASET_PNG_PATH = 'D:/game/G78/AI/data/sketch_data/png/'
    with open(os.path.join(DATASET_PNG_PATH, 'filelist.txt')) as f:
        total_count = 0
        while True:
            target_sketch_image = f.readline().strip('\n')
            if not target_sketch_image:
                break
            image_path = os.path.join(DATASET_PNG_PATH, target_sketch_image)
            image_array = scipy.misc.imread(image_path)
            image_array = scipy.misc.imresize(image_array, size=(400, 400))
            N_ = 50
            E_ = GetStrokePointsHarris(image_array, N_)
            print('Count of detected stroke points is %r/%r' % (len(E_), N_))
            plt.imshow(image_array, cmap='gray')
            for x_, y_, e_ in E_:
                plt.scatter(y_, x_, marker='x', c='g')
            plt.show()
            break

