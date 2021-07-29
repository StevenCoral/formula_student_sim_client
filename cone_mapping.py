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


def process_camera(tf_to_cam, vector, camera, image, tracked_cone):
    vector_camera = np.matmul(tf_to_cam, vector)[:3]
    hsv_image, hsv_success = camera.get_cropped_hsv(image, vector_camera)
    cone_color = tracked_cone.determine_color(hsv_image) if hsv_success else 0
    # if hsv_success and cone_color == 0:
    #     h_range, w_range, midpoint = camera.generate_cropping_indices(vector_camera)
    #     image = cv2.rectangle(image, [w_range[1], h_range[1]], [w_range[0], h_range[0]], (0, 0, 255), 1)
    #     cv2.imshow('f', image)
    #     cv2.waitKey()
    #     a=5
        # hsv_image, hsv_success = camera.get_cropped_hsv(image, vector_camera)
    return cone_color


def mapping_loop(client):
    image_dest = 'D:\\MscProject\\BGR_client\\images'
    data_dest = 'D:\\MscProject\\BGR_client\\test'
    save_data = False

    # Constant transform matrices:
    # Notating A_to_B means that taking a vector in frame A and left-multiplying by the matrix
    # will result in the same point represented in frame B, even though the definition of the deltas
    # within the parentheses describe the transformation from B to A.
    lidar_pos = [2, 0, -0.1]
    lidar_rot = [0, 0, 0]
    left_cam = camera_utils.AirsimCamera(640, 360, 70, [2, -0.5, -0.5], [-40.0, -10.0, 0])
    right_cam = camera_utils.AirsimCamera(640, 360, 70, [2, 0.5, -0.5], [40.0, -10.0, 0])
    lidar_to_vehicle = spatial_utils.tf_matrix_from_airsim_pose(lidar_pos, lidar_rot)
    left_cam_to_vehicle = left_cam.tf_matrix
    right_cam_to_vehicle = right_cam.tf_matrix
    lidar_to_left_cam = np.matmul(np.linalg.inv(left_cam_to_vehicle), lidar_to_vehicle)
    lidar_to_right_cam = np.matmul(np.linalg.inv(right_cam_to_vehicle), lidar_to_vehicle)

    spatial_utils.set_airsim_pose(client, [0.0, 0.0], [90.0, 0, 0])
    time.sleep(1.0)

    car_controls = airsim.CarControls()
    car_controls.throttle = 0.2
    client.setCarControls(car_controls)

    loop_trigger = False
    leaving_distance = 10.0
    entering_distance = 4.0

    tracked_cones = []
    pursuit_points = [np.array([0.0, 0.0, 0.0])]
    min_lookahead_distance = 2.0
    max_lookahead_distance = 6.0
    start_time = time.time()
    last_iteration = start_time
    sample_time = 0.1

    while last_iteration - start_time < 300:
        delta_time = time.time() - last_iteration
        if delta_time > sample_time:
            last_iteration = time.time()
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

            # To minimize discrepancy between data sources, all acquisitions must be made before processing:
            responses = client.simGetImages([airsim.ImageRequest("LeftCam", 0, False, False),
                                             airsim.ImageRequest("RightCam", 0, False, False)])
            point_cloud = aggregate_detections(client)  # Airsim's lidar implementation requires aggregation

            # Save the images in memory
            left_image = camera_utils.get_bgr_image(responses[0])
            right_image = camera_utils.get_bgr_image(responses[1])

            # DBSCAN filtering is done on the sensor-frame
            filtered_pc = dbscan_utils.filter_cloud(point_cloud, 3.0, 10.0, -0.5, 1.0)

            # Only if SOME clusters were found:
            if filtered_pc.size > 0:
                # Cluster centroids, filter them by extent and then sort them by ascending distance:
                db = DBSCAN(eps=0.3, min_samples=3).fit(filtered_pc)
                curr_segments, curr_centroids, curr_labels = dbscan_utils.collate_segmentation(db, 1.0)
                curr_centroids.sort(key=lambda x: np.linalg.norm(x))
                curr_tracked = np.ndarray(shape=(0, 3))

                # camera_utils.save_img(responses[0], os.path.join(image_dest, 'left_' + str(idx) + '.png'))
                # camera_utils.save_img(responses[1], os.path.join(image_dest, 'right_' + str(idx) + '.png'))

                # Go through the DBSCAN centroids of the current frame:
                for centroid_airsim in curr_centroids:
                    centroid_eng, dump = spatial_utils.convert_eng_airsim(centroid_airsim, [0, 0, 0])
                    centroid_lidar = np.append(centroid_eng, 1)
                    centroid_global = np.matmul(lidar_to_map, centroid_lidar)[:3]
                    # We must track yellow and blue cones within the common (global) frame of reference.

                    centroid_exists = False
                    # Compare them against all the known tracked objects:
                    for curr_cone in tracked_cones:
                        if curr_cone.check_proximity(centroid_global):
                            centroid_exists = True
                            curr_cone.process_detection(centroid_global)
                            if curr_cone.active:
                                # Estimate color only for active cones, within camera frustum.
                                # Color estimation is done in the camera frame of reference.
                                centroid_vehicle = np.matmul(lidar_to_vehicle, centroid_lidar)[:3]
                                curr_tracked = np.append(curr_tracked, centroid_vehicle.reshape(1, 3), axis=0)
                                if centroid_vehicle[1] > 0:  # Positive y means left side.
                                    cone_color = process_camera(lidar_to_left_cam,
                                                                centroid_lidar,
                                                                left_cam,
                                                                left_image,
                                                                curr_cone)
                                else:
                                    cone_color = process_camera(lidar_to_right_cam,
                                                                centroid_lidar,
                                                                right_cam,
                                                                right_image,
                                                                curr_cone)
                            break
                    # If no centroid is close enough to an existing one, create a new tracker instance:
                    if not centroid_exists:
                        new_centroid = tracker_utils.ConeTracker(centroid_global)
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
                            last_pursuit = (last_blue + last_yellow) / 2.0
                            pursuit_diff = np.linalg.norm(pursuit_points[-1] - last_pursuit)
                            # Only add the current point to the list if it is not the same as the last one,
                            # and if the discovered cones are closer than 1-diagonal of cones:
                            if pursuit_diff > 1.0:
                                pursuit_points.append(last_pursuit)
                            break

            # Scanning pursuit points from newest (and farthest) backwards.
            # If the point is farther than the max lookahead distance, keep scanning.
            # If we reached a point that is too close, keep moving straight until (hopefully) a new one gets within range.
            if len(pursuit_points) > 2:
                for point in reversed(pursuit_points):
                    # Compute pursuit point in vehicle coordinates:
                    pursuit_map = np.append(point, 1)
                    pursuit_vehicle = np.matmul(map_to_vehicle, pursuit_map)[:3]
                    pursuit_distance = np.linalg.norm(pursuit_vehicle)
                    if pursuit_distance < max_lookahead_distance:
                        break
                if pursuit_distance > min_lookahead_distance:
                    desired_steer = -0.5 * np.arctan(pursuit_vehicle[1] / pursuit_vehicle[0])
                else:
                    desired_steer = 0.0
            else:
                desired_steer = 0.0  # Move straight if we haven't picked up any points yet.

            desired_steer = np.clip(desired_steer, -0.2, 0.2)
            car_controls.steering = desired_steer
            client.setCarControls(car_controls)
            print(delta_time)

        else:
            time.sleep(0.005)

    if save_data:
        tracked_objects = {'cones': tracked_cones, 'pursuit': pursuit_points}
        with open(os.path.join(data_dest, 'tracker_session.pickle'), 'wb') as pickle_file:
            pickle.dump(tracked_objects, pickle_file)
        print('pickle saved')

    return tracked_cones


if __name__ == '__main__':
    airsim_client = airsim.CarClient()
    airsim_client.confirmConnection()
    airsim_client.enableApiControl(True)
    dump = mapping_loop(airsim_client)

    # Done! stop vehicle:
    car_controls = airsim_client.getCarControls()
    car_controls.throttle = 0.0
    car_controls.brake = 1.0
    airsim_client.setCarControls(car_controls)
