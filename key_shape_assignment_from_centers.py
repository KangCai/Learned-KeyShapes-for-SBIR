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
    # assign, assign_reversed = Assignment()
    # print(assign)
    # for i_, c in enumerate(assign_reversed):
    #     print(i_, len(c))
    DrawCluster()
# 0 215
# 1 411
# 2 305
# 3 249
# 4 380
# 5 491
# 6 490
# 7 1078
# 8 5588
# 9 771
# 10 333
# 11 1121
# 12 395
# 13 526
# 14 464
# 15 361
# 16 304
# 17 197
# 18 229
# 19 114
# 20 536
# 21 536
# 22 460
# 23 459
# 24 871
# 25 2778
# 26 690
# 27 372
# 28 414
# 29 560
# 30 204
# 31 700
# 32 469
# 33 350
# 34 1038
# 35 80
# 36 540
# 37 562
# 38 637
# 39 1232
# 40 351
# 41 532
# 42 203
# 43 387
# 44 609
# 45 700
# 46 1276
# 47 335
# 48 496
# 49 528
# 50 1127
# 51 267
# 52 1006
# 53 456
# 54 359
# 55 375
# 56 349
# 57 489
# 58 189
# 59 1250
# 60 507
# 61 348
# 62 451
# 63 138
# 64 924
# 65 618
# 66 480
# 67 751
# 68 426
# 69 649
# 70 2310
# 71 701
# 72 408
# 73 337
# 74 487
# 75 513
# 76 1156
# 77 449
# 78 445
# 79 607
# 80 1329
# 81 1048
# 82 554
# 83 505
# 84 3594
# 85 959
# 86 274
# 87 318
# 88 370
# 89 443
# 90 1955
# 91 305
# 92 594
# 93 629
# 94 545
# 95 999
# 96 457
# 97 402
# 98 473
# 99 362
# 100 1015
# 101 416
# 102 262
# 103 273
# 104 250
# 105 761
# 106 413
# 107 166
# 108 415
# 109 198
# 110 340
# 111 602
# 112 430
# 113 483
# 114 448
# 115 560
# 116 1502
# 117 562
# 118 254
# 119 552
# 120 393
# 121 190
# 122 391
# 123 414
# 124 366
# 125 190
# 126 475
# 127 434
# 128 801
# 129 428
# 130 269
# 131 464
# 132 340
# 133 259
# 134 221
# 135 580
# 136 535
# 137 511
# 138 456
# 139 670
# 140 389
# 141 923
# 142 347
# 143 524
# 144 358
# 145 3293
# 146 434
# 147 329
# 148 190
# 149 299