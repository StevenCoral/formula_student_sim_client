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
        lidar_data = airsim_client.getLidarData()
        pointcloud = np.append(pointcloud, np.array(lidar_data.point_cloud, dtype=np.dtype('f4')))
    return np.reshape(pointcloud, (int(pointcloud.shape[0] / 3), 3))


def mapping_loop():
    image_dest = 'D:\\MscProject\\BGR_client\\images'
    save_data = True

    # Constant transform matrices:
    # Notating A_to_B means that taking a vector in frame A and left-multiplying by the matrix
    # will result in the same point represented in frame B, even though the definition of the deltas
    # within the parentheses describe the transformation from B to A.

    lidar_pos = [2, 0, -0.1]
    lidar_rot = [0, 0, 0]

    left_cam = camera_utils.AirsimCamera(640, 360, 80, [2, -0.5, -0.5], [-40.0, -10.0, 0])
    right_cam = camera_utils.AirsimCamera(640, 360, 80, [2, 0.5, -0.5], [40.0, -10.0, 0])

    lidar_to_vehicle = spatial_utils.tf_matrix_from_airsim_pose(lidar_pos, lidar_rot)
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

    loop_trigger = False
    leaving_distance = 10.0
    entering_distance = 6.0

    tracked_cones = []
    pursuit_points = [np.array([0.0, 0.0, 0.0])]
    lookahead_distance = 6.0
    start_time = time.time()

    while time.time() - start_time < 300:
        tic = time.time()
        vehicle_pose = client.simGetVehiclePose()
        vehicle_to_map = spatial_utils.tf_matrix_from_airsim_object(vehicle_pose)
        map_to_vehicle = np.linalg.inv(vehicle_to_map)
        lidar_to_map = np.matmul(vehicle_to_map, lidar_to_vehicle)

        distance_from_start = np.linalg.norm(vehicle_to_map[0:2, 3])
        if not loop_trigger:
            if distance_from_start > leaving_distance:
                loop_trigger = True
        else:
            if distance_from_start < entering_distance:
                break

        # DBSCAN filtering is done on the sensor-frame
        point_cloud = aggregate_detections(client)  # Airsim's stupid lidar implementation requires aggregation
        filtered_pc = dbscan_utils.filter_cloud(point_cloud, 3.0, 10.0, -0.5, 1.0)

        # Only is SOME clusters were found:
        if filtered_pc.size > 0:
            db = DBSCAN(eps=0.3, min_samples=3).fit(filtered_pc)
            curr_segments, curr_centroids, curr_labels = dbscan_utils.collate_segmentation(db, 1.0)
            # Sort centroids by distance from vehicle for target point extraction later.
            curr_centroids.sort(key=lambda x: np.linalg.norm(x))

            responses = client.simGetImages([airsim.ImageRequest("LeftCam", 0, False, False),
                                             airsim.ImageRequest("RightCam", 0, False, False)])

            # camera_utils.save_img(responses[0], os.path.join(image_dest, 'left_' + str(idx) + '.png'))
            # camera_utils.save_img(responses[1], os.path.join(image_dest, 'right_' + str(idx) + '.png'))
            left_image = camera_utils.get_bgr_image(responses[0])
            right_image = camera_utils.get_bgr_image(responses[1])

            # Go through the DBSCAN centroids of the current frame:
            for centroid_airsim in curr_centroids:
                centroid_eng, dump = spatial_utils.convert_eng_airsim(centroid_airsim, [0, 0, 0])
                centroid_local = np.append(centroid_eng, 1)
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
                    # If no centroid is close enough to an existing one, create a new tracker instance:
                    if not centroid_exists:
                        new_centroid = tracker_utils.ConeTracker(centroid_global)
                        new_centroid.determine_color(hsv_image)
                        tracked_cones.append(new_centroid)

            # Create pursuit points:
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
                        # Only add the current point to the list if it is not the same as the last one:
                        if np.linalg.norm(pursuit_points[-1] - last_pursuit) > 1.0:
                            pursuit_points.append(last_pursuit)
                        break

        if len(pursuit_points) > 2:
            for point in reversed(pursuit_points):
                # Compute pursuit point in vehicle coordinates:
                pursuit_map = np.append(point, 1)
                pursuit_vehicle = np.matmul(map_to_vehicle, pursuit_map)[:3]
                if np.linalg.norm(pursuit_vehicle) > lookahead_distance:
                    pursuit_candidate = pursuit_vehicle
                else:
                    break
            desired_steer = -0.5 * np.arctan(pursuit_vehicle[1] / pursuit_vehicle[0])

        else:
            desired_steer = 0.0  # Move straight if we haven't picked up any points yet.

        # if len(pursuit_points) > 2:  # Two steps from the list's end yields a closer lookahead distance
        #     # Compute pursuit point in vehicle coordinates:
        #     pursuit_map = np.append(pursuit_points[-2], 1)
        #     pursuit_vehicle = np.matmul(map_to_vehicle, pursuit_map)[:3]
        #     desired_steer = -0.5 * np.arctan(pursuit_vehicle[1] / pursuit_vehicle[0])
        # else:
        #     desired_steer = 0.0  # Move straight if we haven't picked up any points yet.

        car_controls.steering = desired_steer
        client.setCarControls(car_controls)

        toc = time.time()
        print(toc - start_time)
        time.sleep(0.05)

    if save_data:
        tracked_objects = {'cones': tracked_cones, 'pursuit': pursuit_points}
        with open('tracker_session.pickle', 'wb') as pickle_file:
            pickle.dump(tracked_objects, pickle_file)
        print('pickle saved')

    return pursuit_points


if __name__ == '__main__':
    useless = mapping_loop()
