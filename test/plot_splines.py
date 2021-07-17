import numpy as np
import pickle
from matplotlib import pyplot as plt
import time
from spline_utils import PathSpline

with open('car_data.pickle', 'rb') as handle:
    car_data = pickle.load(handle)

with open('control_data.pickle', 'rb') as handle:
    control_data = pickle.load(handle)

# x = np.array([10.00, 10.00, 10.00, 10.00, 10.00, 6.00, -6.00, -18.00, -23.00, -23.00, -17.00, 0.0, 8.00])
# y = np.array([20.00, 10.00, -10.00, -40.00, -60.00, -73.00, -78.00, -70.00, -38.00, 10.00, 30.00, 31.00, 27.00])
#
#
# tangent_x = control_data[:, 4]
# tangent_y = control_data[:, 5]
# dirs = np.arctan2(tangent_y, tangent_x)
# thetas = np.rad2deg(control_data[:, 6:])
# steer = thetas[:, 0] + thetas[:, 1]
#
# my_spline = PathSpline(x, y)
# my_spline.generate_spline(0.1)
#
#
# fig, ax = plt.subplots(1, 1)
# ax.grid(True)
# ax.axis('equal')
# ax.plot(car_data[:, 0] + 10.0, car_data[:, 1] + 20.0, color='blue', linewidth=0.2)
# ax.plot(my_spline.xi, my_spline.yi, color='red', linewidth=0.2)
# fig.show()

# for curr_row in range(car_data.shape[0]):
#     ax.plot(car_data[:, 0], car_data[:, 1], '.b')
#     ax.plot(car_data[curr_row, 0], car_data[curr_row, 1], 'or')
#     dir_x = [car_data[curr_row, 0],
#              car_data[curr_row, 0] + car_data[curr_row, 3] * np.cos(car_data[curr_row, 2])]
#     dir_y = [car_data[curr_row, 1],
#              car_data[curr_row, 1] + car_data[curr_row, 3] * np.sin(car_data[curr_row, 2])]
#     ax.plot(dir_x, dir_y, 'or')
#     fig.show()
#     ax.cla()

fig, ax = plt.subplots(1, 1)
ax.plot(car_data[:, 3], '.b')
fig.show()
a=5
pass