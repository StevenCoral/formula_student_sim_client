import numpy as np
import pickle
from matplotlib import pyplot as plt
import time
from spline_utils import PathSpline

with open('tracker_session.pickle', 'rb') as handle:
    tracker_data = pickle.load(handle)

left_points = np.ndarray(shape=(0, 2))
for tracked_obj in tracker_data['left']:
    if tracked_obj.active:
        left_points = np.append(left_points, tracked_obj.position[0:2].reshape(1, 2), axis=0)

right_points = np.ndarray(shape=(0, 2))
for tracked_obj in tracker_data['right']:
    if tracked_obj.active:
        right_points = np.append(right_points, tracked_obj.position[0:2].reshape(1, 2), axis=0)

fig, ax = plt.subplots(1, 1)
ax.plot(right_points[:, 0], right_points[:, 1], 'or')
ax.plot(left_points[:, 0], left_points[:, 1], 'ob')
ax.grid(True)
ax.axis('equal')
# plt.xlim([-20, 20])
# plt.ylim([0, 40])
fig.show()
a = 5
