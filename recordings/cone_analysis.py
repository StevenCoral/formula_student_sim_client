import numpy as np
import pickle
import csv
import tracker_utils
import spline_utils
from matplotlib import pyplot as plt
import os
import sys
import time

COLOR_UNKNOWN = tracker_utils.ConeTracker.COLOR_UNKNOWN
COLOR_BLUE = tracker_utils.ConeTracker.COLOR_BLUE
COLOR_YELLOW = tracker_utils.ConeTracker.COLOR_YELLOW


def count_detections(trackers_list, expected_color):
    unknown_count = 0
    misclass_count = 0
    detection_counts = np.array([])
    for curr_tracker in trackers_list:
        detection_counts = np.append(detection_counts, len(curr_tracker.color_history)-1)
        for curr_class in curr_tracker.color_history[1:]:
            if curr_class != expected_color:
                if curr_class == 0:
                    unknown_count += 1
                else:
                    misclass_count += 1
    return unknown_count, misclass_count, detection_counts


def calculate_errors(trackers_list, cones_list):
    squared_errors = np.array([])
    for curr_tracker in trackers_list:
        # Find closest cone in the list:
        differences = cones_list - curr_tracker.position[:2]
        distances = np.linalg.norm(differences, axis=1)
        idx = np.argmin(distances)
        for curr_pos in curr_tracker.detections:
            diff = cones_list[idx, :] - curr_pos[:2]
            dist = np.linalg.norm(diff)
            squared_errors = np.append(squared_errors, dist)
    return squared_errors


single_run = True
player_start = np.array([12.1, 18.7])

# Begin cone data extraction:
true_cones = np.genfromtxt('true_cone_locations.csv', delimiter=',')
true_cones = true_cones[2:, :]  # Taking out titles and false [0,3] instances
true_yellows = true_cones[:, 0:2] - player_start
true_yellows[:, 1] *= -1
true_blues = true_cones[:, 2:] - player_start
true_blues[:, 1] *= -1

if single_run:
    data_folder = os.path.join(os.getcwd(), 'recording1')
    with open(os.path.join(data_folder, 'mapping_session.pickle'), 'rb') as handle:
        mapping_data = pickle.load(handle)
    cone_data = mapping_data['cones']
    car_data = np.genfromtxt(os.path.join(data_folder, 'car_data.csv'), delimiter=',')
    car_data = car_data[1:, :]
else:
    # Loop over folders and aggregate cones:
    cone_data = []
    for aggregation in range(10):
        data_folder = os.path.join(os.getcwd(), 'recording' + str(aggregation))
        with open(os.path.join(data_folder, 'mapping_session.pickle'), 'rb') as handle:
            mapping_data = pickle.load(handle)
        cone_data = cone_data + mapping_data['cones']

csv_folder = os.path.join(os.getcwd(), 'csv')
inactive_cones = []
yellow_cones = []
blue_cones = []
unknown_cones = []
blue_points = np.ndarray(shape=(0, 2))
yellow_points = np.ndarray(shape=(0, 2))
unknown_points = np.ndarray(shape=(0, 2))
inactive_points = np.ndarray(shape=(0, 2))

# Separate detected cones into blues, yellows and unknowns:
for tracked_obj in cone_data:
    if tracked_obj.active:
        if tracked_obj.color == tracker_utils.ConeTracker.COLOR_BLUE:
            blue_cones.append(tracked_obj)
            blue_points = np.append(blue_points, tracked_obj.position[0:2].reshape(1, 2), axis=0)
        elif tracked_obj.color == tracker_utils.ConeTracker.COLOR_YELLOW:
            yellow_cones.append(tracked_obj)
            yellow_points = np.append(yellow_points, tracked_obj.position[0:2].reshape(1, 2), axis=0)
        elif tracked_obj.color == tracker_utils.ConeTracker.COLOR_UNKNOWN:
            unknown_cones.append(tracked_obj)
            unknown_points = np.append(unknown_points, tracked_obj.position[0:2].reshape(1, 2), axis=0)
        else:
            print('Nonexistent color!!')
    else:
        inactive_cones.append(tracked_obj)
        np.append(inactive_points, tracked_obj.position[0:2].reshape(1, 2), axis=0)

if single_run:
    # Begin path data extraction:
    min_length = min(blue_points.shape[0], yellow_points.shape[0])
    track_points = (blue_points[:min_length, :] + yellow_points[:min_length, :]) / 2
    # raw_spline = spline_utils.PathSpline(track_points[:, 0], track_points[:, 1])
    # raw_spline.generate_spline(amount=0.1, meters=True, smoothing=0)
    smoothed_spline = spline_utils.PathSpline(track_points[::2, 0], track_points[::2, 1])
    smoothed_spline.generate_spline(amount=0.1, meters=True, smoothing=1)

    detected_cones = np.append(blue_points[:min_length, :], yellow_points[:min_length, :], axis=1)
    with open(os.path.join(csv_folder, 'detected_cone_locations.csv'), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['yellow x', 'yellow y', 'blue x', 'blue y'])
        writer.writerows(detected_cones)

    pursuit_points = np.asarray(mapping_data['pursuit'])
    with open(os.path.join(csv_folder, 'pursuit_points.csv'), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['x', 'y', 'z'])
        writer.writerows(pursuit_points)

    spline_length = smoothed_spline.xi.size
    spline_points = np.append(smoothed_spline.xi.reshape(spline_length, 1),
                              smoothed_spline.yi.reshape(spline_length, 1),
                              axis=1)
    spline_points = np.append(spline_points,
                              smoothed_spline.curvature.reshape(spline_length, 1),
                              axis=1)
    with open(os.path.join(csv_folder, 'spline_points.csv'), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['x', 'y', 'curvature'])
        writer.writerows(spline_points)

    car_positions = car_data[:, :2]
    forward_idx = int(np.around(5.0 / smoothed_spline.meters_per_index))
    lookahead_indices = np.array([])
    for curr_pos in car_positions:
        closest_idx, dump1, dump2 = smoothed_spline.find_closest_point(curr_pos)
        lookahead_idx = closest_idx + forward_idx
        if lookahead_idx >= smoothed_spline.array_length:
            lookahead_idx = lookahead_idx - smoothed_spline.array_length
        lookahead_indices = np.append(lookahead_indices, lookahead_idx)

    with open(os.path.join(csv_folder, 'lookahead_indices.csv'), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['index'])
        writer.writerows(lookahead_indices.reshape(lookahead_indices.size, 1))

    print('saved csv data')

else:
    # Gather cone tracker statistics:
    blue_unknown, blue_misclass, blue_dets = count_detections(blue_cones, COLOR_BLUE)
    blue_errors = calculate_errors(blue_cones, true_blues)
    yellow_unknown, yellow_misclass, yellow_dets = count_detections(yellow_cones, COLOR_YELLOW)
    yellow_errors = calculate_errors(yellow_cones, true_yellows)

    blue_dets = np.array([blue_dets])
    with open(os.path.join(csv_folder, 'blue_detections.csv'), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['detection_count'])
        writer.writerows(blue_dets)

    with open(os.path.join(csv_folder, 'blue_errors.csv'), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['detection_errors'])
        writer.writerows(blue_errors.reshape(blue_errors.size, 1))

    yellow_dets = np.array([yellow_dets])
    with open(os.path.join(csv_folder, 'yellow_detections.csv'), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['detection_count'])
        writer.writerows(yellow_dets)
    print('saved csv data')

    with open(os.path.join(csv_folder, 'yellow_errors.csv'), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['detection_errors'])
        writer.writerows(yellow_errors.reshape(yellow_errors.size, 1))

    # blue_hist = np.histogram(blue_errors, bins=100)
    # plt.hist(blue_dets, 20)
    # plt.waitforbuttonpress()

a = 5

# fig, ax = plt.subplots(1, 1)
# ax.grid(True)
# ax.axis('equal')
#
# ax.plot(smoothed_spline.xi, smoothed_spline.yi, color='blue')
# ax.plot(raw_spline.xi, raw_spline.yi, color='red')

# ax.plot(blue_points[:, 0], blue_points[:, 1], '.b')
# ax.plot(yellow_points[:, 0], yellow_points[:, 1], '.y')
# ax.plot(unknown_points[:, 0], unknown_points[:, 1], '.k')
#
# ax.plot(true_blues[:, 0], true_blues[:, 1], '.g')
# ax.plot(true_yellows[:, 0], true_yellows[:, 1], '.r')
#
# fig.show()
# plt.waitforbuttonpress()

