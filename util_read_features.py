# coding=utf-8

import os
import consts
import numpy as np

def ReadArray():
    feature_list, file_list = [], []
    for c in range(consts.COUNT_CLIP):
        save_feature_file_path = os.path.join(consts.SAVE_PATCH_CLUSTER_CENTER_PATH, consts.FEATURE_SAVE_NAME + str(c) + '.txt')
        print('Reading file %r ...' % (save_feature_file_path,))
        with open(save_feature_file_path) as f:
            a = f.read()
            tmp_feature_list, tmp_file_list = [i[0] for i in eval(a)], [i[1] for i in eval(a)]
            feature_list.extend(tmp_feature_list)
            file_list.extend(tmp_file_list)
        print('Done')
    return np.array(feature_list), file_list