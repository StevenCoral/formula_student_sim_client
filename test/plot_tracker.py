import numpy as np
import pickle
from matplotlib import pyplot as plt
import time
from spline_utils import PathSpline

with open('dbscan_session.pickle', 'rb') as handle:
    dbscan_data = pickle.load(handle)

all_segments = dbscan_data['all_segments']
all_centroids = dbscan_data['all_centroids']
del all_segments
del dbscan_data

fig, ax = plt.subplots(1, 1)
for idx in range(len(all_centroids)):
    if all_centroids[idx]:
        curr_segments = np.ndarray(shape=(0, 3))
        curr_centroids = np.ndarray(shape=(0, 3))
        for curr_frame in all_centroids[idx]:
            # debug_seg = all_segments[idx][curr_frame]
            # curr_segments = np.append(curr_segments, all_segments[idx][curr_frame], axis=0)
            # debug_cent = all_centroids[idx][curr_frame].reshape(1, 3)
            curr_centroids = np.append(curr_centroids, all_centroids[idx][curr_frame].reshape(1, 3), axis=0)

        # ax.plot(curr_segments[:, 0], curr_segments[:, 1], '.b')
        ax.plot(curr_centroids[:, 0], curr_centroids[:, 1], 'or')
        ax.grid(True)
        plt.xlim([0, 20])
        plt.ylim([-5, 5])
        fig.show()
        a = 5
        ax.clear()

