"""
written by lily
generator data set from original labels, where each row is formatted as:
start_timestamp, end_timestamp, channel, class, difficult

As for this compared method which doesn't distinguish different events of the same appliance,
we only use start_timestamp and end_timestamp to extract transient serial (9 point), that's all
"""


import numpy as np


def get_main_data(path):
    data = np.loadtxt(path, usecols=[0, 1])
    labels = np.loadtxt(path)
    results = []
    for label in labels:
        start_idx = np.argwhere(data[:, 0] == label[0])[0][0]
        end_idx = np.argwhere(data[:, 0] == label[1])[0][0]
        need = 9 - (end_idx + 1 - start_idx)
        start_idx = start_idx - need // 2
        end_idx = end_idx + (need + 1) // 2
        assert end_idx + 1 - start_idx == 9, f"not 9"
        results.append(data[start_idx, end_idx+1, 1] + )
