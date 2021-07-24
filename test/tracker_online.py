#!/usr/bin/env python
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import time
import pickle
import csv
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import airsim
import dbscan_utils
import spatial_utils
import tracker_utils
import camera_utils
import os
import cv2


def aggregate_detections(airsim_client, iterations=2):
    pointcloud = np.array([])
    for curr_iter in range(iterations):
        lidarData = airsim_client.getLidarData()
        pointcloud = np.append(pointcloud, np.array(lidarData.point_cloud, dtype=np.dtype('f4')))
    return np.reshape(pointcloud, (int(pointcloud.shape[0] / 3), 3))


def compare_trackers(centroid_pos, tracker_list):
    centroid_exists = False
    # Compare them against all the known tracked objects:
    for centroid in tracker_list:
        if centroid.check_proximity(centroid_pos):
            centroid_exists = True
            centroid.process_detection(centroid_pos)

    # If it is not close enough to an existing point, create a new tracker instance:
    if not centroid_exists:
        tracker_list.append(tracker_utils.ConeTracker(centroid_pos))


if __name__ == '__main__':

    # Airsim is stupid, always spawns at zero. Must compensate using "playerstart" in unreal:
    starting_x = 10.0
    starting_y = 20.0
    image_dest = 'D:\\MscProject\\BGR_client\\images'

    # Constant transform matrices:
    # Notating A_to_B means that taking a vector in frame A and left-multiplying by the matrix
    # will result in the same point represented in frame B, even though the definition of the deltas
    # within the parentheses describe the transformation from B to A.

    left_cam = camera_utils.AirsimCamera(640, 360, 90, [2, -0.5, -0.5], [-40.0, -10.0, 0])
    right_cam = camera_utils.AirsimCamera(640, 360, 90, [2, 0.5, -0.5], [40.0, -10.0, 0])
    lidar_pos = [2, 0, -0.1]
    lidar_rot = [0, 0, 0]

    lidar_to_vehicle = spatial_utils.tf_matrix_from_airsim_pose(lidar_pos, lidar_rot)
    # left_cam_to_vehicle = spatial_utils.tf_matrix_from_airsim_pose([2, -0.5, -0.5], [-40.0, -10.0, 0])
    # right_cam_to_vehicle = spatial_utils.tf_matrix_from_airsim_pose([2, 0.5, -0.5], [40.0, -10.0, 0])
    left_cam_to_vehicle = left_cam.tf_matrix
    right_cam_to_vehicle = right_cam.tf_matrix
    lidar_to_left_cam = np.matmul(np.linalg.inv(left_cam_to_vehicle), lidar_to_vehicle)
    lidar_to_right_cam = np.matmul(np.linalg.inv(right_cam_to_vehicle), lidar_to_vehicle)

    # connect to the AirSim simulator
    client = airsim.CarClient()
    client.confirmConnection()
    spatial_utils.set_airsim_pose(client, [0.0, 0.0], [90.0, 0, 0])
    time.sleep(1.0)

    client.enableApiControl(True)
    car_controls = airsim.CarControls()
    car_controls.throttle = 0.1
    client.setCarControls(car_controls)

    right_trackers = []
    left_trackers = []
    right_cam_centroids = []
    left_cam_centroids = []

    tracked_cones = []
    pursuit_points = [np.array([0.0, 0.0, 0.0])]
    start_time = time.time()
    idx = 0
    while time.time() - start_time < 120:
        tic = time.time()
        vehicle_pose = client.simGetVehiclePose()
        vehicle_to_map = spatial_utils.tf_matrix_from_airsim_object(vehicle_pose)
        map_to_vehicle = np.linalg.inv(vehicle_to_map)

        lidar_to_map = np.matmul(vehicle_to_map, lidar_to_vehicle)

        # DBSCAN filtering is done on the sensor-frame
        point_cloud = aggregate_detections(client)  # Airsim's stupid lidar implementation requires aggregation
        filtered_pc = dbscan_utils.filter_cloud(point_cloud, 3.0, 10.0, -0.5, 1.0)

        # print(idx)
        if filtered_pc.size > 0:
            db = DBSCAN(eps=0.3, min_samples=3).fit(filtered_pc)
            curr_segments, curr_centroids, curr_labels = dbscan_utils.collate_segmentation(db, 1.0)
            # Sort centroids by distance from vehicle for goal point extraction later.
            curr_centroids.sort(key=lambda x: np.linalg.norm(x))

            responses = client.simGetImages([airsim.ImageRequest("LeftCam", 0, False, False),
                                             airsim.ImageRequest("RightCam", 0, False, False)])

            # camera_utils.save_img(responses[0], os.path.join(image_dest, 'left_' + str(idx) + '.png'))
            # camera_utils.save_img(responses[1], os.path.join(image_dest, 'right_' + str(idx) + '.png'))
            left_image = camera_utils.get_bgr_image(responses[0])
            right_image = camera_utils.get_bgr_image(responses[1])

            # Go through the DBSCAN centroids of the current frame:
            for centroid_airsim in curr_centroids:
                # centroid_exists = False
                centroid_eng, dump = spatial_utils.convert_eng_airsim(centroid_airsim, [0, 0, 0])
                centroid_local = np.append(centroid_eng, 1)
                centroid_local[2] -= 0.05  # Getting closer to the cone's middle height.
                centroid_global = np.matmul(lidar_to_map, centroid_local)[:3]
                centroid_vehicle = np.matmul(lidar_to_vehicle, centroid_local)[:3]

                if centroid_vehicle[1] > 0:  # Positive y means left side.
                    centroid_camera = np.matmul(lidar_to_left_cam, centroid_local)[:3]
                    hsv_image, hsv_success = left_cam.get_cropped_hsv(left_image, centroid_camera)
                else:
                    centroid_camera = np.matmul(lidar_to_right_cam, centroid_local)[:3]
                    hsv_image, hsv_success = right_cam.get_cropped_hsv(right_image, centroid_camera)

                if hsv_success:  # Only if cone is within camera frustum!
                    centroid_exists = False
                    # Compare them against all the known tracked objects:
                    for centroid in tracked_cones:
                        # We must track yellow and blue cones within the common (global) frame of reference:
                        if centroid.check_proximity(centroid_global):
                            centroid_exists = True
                            centroid.process_detection(centroid_global)
                            centroid.determine_color(hsv_image)
                            break
                    # If it is not close enough to an existing point, create a new tracker instance:
                    if not centroid_exists:
                        new_centroid = tracker_utils.ConeTracker(centroid_global)
                        new_centroid.determine_color(hsv_image)
                        tracked_cones.append(new_centroid)

            # Create following points:
            last_blue = None
            last_yellow = None
            for curr_cone in reversed(tracked_cones):
                if curr_cone.active:
                    if last_blue is None and curr_cone.color == curr_cone.COLOR_BLUE:
                        last_blue = curr_cone.position
                    if last_yellow is None and curr_cone.color == curr_cone.COLOR_YELLOW:
                        last_yellow = curr_cone.position
                    if last_blue is not None and last_yellow is not None:
                        last_pursuit = (last_blue + last_yellow) / 2
                        # Only add to the list if it is a "new point"
                        if np.linalg.norm(pursuit_points[-1] - last_pursuit) > 1.0:
                            pursuit_points.append(last_pursuit)
                        break

        if len(pursuit_points) > 1:
            pursuit_map = np.append(pursuit_points[-1], 1)
            pursuit_vehicle = np.matmul(map_to_vehicle, pursuit_map)[:3]
            desired_steer = -0.5 * pursuit_vehicle[1] / max(1.0, np.linalg.norm(pursuit_vehicle))
        else:
            desired_steer = 0.0  # Move straight if we havent picked up any points yet.

        car_controls.steering = desired_steer
        client.setCarControls(car_controls)

        idx += 1
        toc = time.time()
        print(idx, toc-tic)
        time.sleep(0.05)

    tracked_objects = {'cones': tracked_cones, 'pursuit': pursuit_points}
    with open('tracker_session.pickle', 'wb') as pickle_file:
        pickle.dump(tracked_objects, pickle_file)

    a=5
    # tracked_centroids = {'left': left_trackers, 'right': right_trackers}
    # camera_centroids = {'left': left_cam_centroids, 'right': right_cam_centroids}
    #
    # with open('tracker_session.pickle', 'wb') as pickle_file:
    #     pickle.dump(tracked_centroids, pickle_file)
    #
    # with open('camera_centroids.pickle', 'wb') as pickle_file:
    #     pickle.dump(camera_centroids, pickle_file)
    # print('saved pickle data')
