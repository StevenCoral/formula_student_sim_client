import numpy as np
import pickle
from matplotlib import pyplot as plt
import time
import tracker_utils
import spline_utils

with open('tracker_session.pickle', 'rb') as handle:
    tracker_data = pickle.load(handle)

problematic = 0
left_points = np.ndarray(shape=(0, 2))
right_points = np.ndarray(shape=(0, 2))
unknown_points = np.ndarray(shape=(0, 2))
for tracked_obj in tracker_data['cones']:
    if tracked_obj.active:
        if tracked_obj.color == tracker_utils.ConeTracker.COLOR_BLUE:
            left_points = np.append(left_points, tracked_obj.position[0:2].reshape(1, 2), axis=0)
        elif tracked_obj.color == tracker_utils.ConeTracker.COLOR_YELLOW:
            right_points = np.append(right_points, tracked_obj.position[0:2].reshape(1, 2), axis=0)
        else:
            point = tracked_obj.position[0:2]
            left_dist = np.linalg.norm(left_points - point, axis=1)
            right_dist = np.linalg.norm(right_points - point, axis=1)
            if np.min(left_dist) < np.min(right_dist):
                left_points = np.append(left_points, point.reshape(1, 2), axis=0)
            else:
                right_points = np.append(right_points, point.reshape(1, 2), axis=0)
            unknown_points = np.append(unknown_points, tracked_obj.position[0:2].reshape(1, 2), axis=0)
    else:
        print('inactive cone at pos:', tracked_obj.position[0:2])

# for point in unknown_points:
#     left_dist = np.linalg.norm(left_points - point, axis=1)
#     right_dist = np.linalg.norm(right_points - point, axis=1)
#     if np.min(left_dist) < np.min(right_dist):
#         left_points = np.append(left_points, point.reshape(1, 2), axis=0)
#     else:
#         right_points = np.append(right_points, point.reshape(1, 2), axis=0)

pursuit_points = np.ndarray(shape=(0, 2))
for tracked_obj in tracker_data['pursuit']:
    pursuit_points = np.append(pursuit_points, tracked_obj[:2].reshape(1, 2), axis=0)

# assuming somewhat equal cone detections on each side!
min_length = min(left_points.shape[0], right_points.shape[0])
track_points = (left_points[:min_length, :] + right_points[:min_length, :]) / 2

my_spline = spline_utils.PathSpline(track_points[::2, 0], track_points[::2, 1])
my_spline.generate_spline(amount=0.1, meters=True, smoothing=1)

fig, ax = plt.subplots(1, 1)
ax.plot(right_points[:, 0], right_points[:, 1], 'or')
ax.plot(left_points[:, 0], left_points[:, 1], 'ob')
# ax.plot(unknown_points[:, 0], unknown_points[:, 1], 'ok')
# ax.plot(track_points[:, 0], track_points[:, 1], 'ok')
ax.plot(my_spline.xi, my_spline.yi)
# ax.plot(pursuit_points[:, 0], pursuit_points[:, 1], 'o')
ax.grid(True)
ax.axis('equal')

fig.show()
a = 5
