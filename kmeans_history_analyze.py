# coding=utf-8

import matplotlib.pyplot as plt

KMEANS_HISTORY_FILE = './kmeans_history.txt'

with open(KMEANS_HISTORY_FILE) as f:
    content = f.read()
    one_line_list = content.split('Kmeans iteration')[1:]
    X, Y = [0], [929890]
    for one_line in one_line_list:
        res_list = one_line.split(' ')
        idx, err_count = int(res_list[1]), int(res_list[2][:-1]) # Total 929890
        X.append(idx+1)
        Y.append(err_count)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.plot(X, Y)
    ax.set_xlim(0, X[-1])
    ax.set_ylim(1, 1e6)
    ax.set_title('Kmeans Error History')
    ax.set_xticks([0, 50, 100, 150, 200, 250, X[-1]])
    # ax.set_yticklabels([0, 50, 100, 150, 200, 250, X[-1]])
    ax.grid()
    plt.show()