import numpy as np
import pickle
from matplotlib import pyplot as plt
import time
from spline_utils import PathSpline

with open('tracker_session.pickle', 'rb') as handle:
    tracker_data = pickle.load(handle)

tracked_points = np.ndarray(shape=(0, 2))
for tracked_obj in tracker_data:
    if tracked_obj.active:
        tracked_points = np.append(tracked_points, tracked_obj.position[0:2].reshape(1, 2), axis=0)

fig, ax = plt.subplots(1, 1)
ax.plot(tracked_points[:, 0], tracked_points[:, 1], 'or')
ax.grid(True)
ax.axis('equal')
# plt.xlim([-20, 20])
# plt.ylim([0, 40])
fig.show()
a = 5
