import numpy as np
import time
import pickle
from sklearn.cluster import DBSCAN
import airsim
import dbscan_utils
import spatial_utils
import tracker_utils
import camera_utils
import path_control
import os
import cv2
import struct


decimation = 30e9  # Used to save an output image every X iterations.


def aggregate_detections(airsim_client, iterations=1):
    pointcloud = np.array([])
    for curr_iter in range(iterations):
        lidar_data = airsim_client.getLidarData()
        pointcloud = np.append(pointcloud, np.array(lidar_data.point_cloud, dtype=np.dtype('f4')))
    return np.reshape(pointcloud, (int(pointcloud.shape[0] / 3), 3))


def process_camera(lidar_to_cam, vector, camera, image, tracked_cone, idx, copy_img):
    vector_camera = np.matmul(lidar_to_cam, vector)[:3]
    hsv_image, hsv_success = camera.get_cropped_hsv(image, vector_camera)
    cone_color = tracked_cone.determine_color(hsv_image) if hsv_success else 0
    global decimation
    if idx > decimation and hsv_success:
        if cone_color == 0:
            bgr_color = (0, 0, 255)
        elif cone_color == 1:
            bgr_color = (255, 0, 0)
        else:
            bgr_color = (0, 255, 255)
        h_range, w_range, midpoint = camera.generate_cropping_indices(vector_camera)
        copy_img = cv2.rectangle(copy_img, [w_range[1], h_range[1]], [w_range[0], h_range[0]], bgr_color, 1)
    return cone_color


def mapping_loop(client):
    global decimation
    image_dest = os.path.join(os.getcwd(), 'images')
    data_dest = os.path.join(os.getcwd(), 'recordings')
    os.makedirs(image_dest, exist_ok=True)
    os.makedirs(data_dest, exist_ok=True)
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

    # Define pure pursuit parameters
    pursuit_follower = path_control.PursuitFollower(2.0, 6.0)
    pursuit_follower.k_steer = 0.5

    # Open access to shared memory blocks:
    shmem_active, shmem_setpoint, shmem_output = path_control.SteeringProcManager.retrieve_shared_memories()

    # Initialize vehicle starting point
    spatial_utils.set_airsim_pose(client, [0.0, 0.0], [90.0, 0, 0])
    time.sleep(1.0)
    car_controls = airsim.CarControls()
    car_controls.throttle = 0.2
    client.setCarControls(car_controls)

    # Initialize loop variables
    tracked_cones = []

    loop_trigger = False
    leaving_distance = 10.0
    entering_distance = 4.0

    start_time = time.perf_counter()
    last_iteration = start_time
    sample_time = 0.1
    execution_time = 0.0

    idx = 0
    save_idx = 0
    while last_iteration - start_time < 300:
        now = time.perf_counter()
        delta_time = now - last_iteration

        if delta_time > sample_time:
            last_iteration = time.perf_counter()
            vehicle_pose = client.simGetVehiclePose()
            vehicle_to_map = spatial_utils.tf_matrix_from_airsim_object(vehicle_pose)
            map_to_vehicle = np.linalg.inv(vehicle_to_map)
            lidar_to_map = np.matmul(vehicle_to_map, lidar_to_vehicle)
            car_state = client.getCarState()
            curr_vel = car_state.speed

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

            lidar_data = client.getLidarData()
            pointcloud = np.array(lidar_data.point_cloud, dtype=np.dtype('f4'))
            pointcloud = pointcloud.reshape((int(pointcloud.shape[0] / 3), 3))

            # Save the images in memory
            left_image = camera_utils.get_bgr_image(responses[0])
            right_image = camera_utils.get_bgr_image(responses[1])

            left_copy = np.copy(left_image)
            right_copy = np.copy(right_image)

            # DBSCAN filtering is done on the sensor-frame
            filtered_pc = dbscan_utils.filter_cloud(pointcloud, 3.0, 8.0, -0.5, 1.0)

            # Only if SOME clusters were found:
            if filtered_pc.size > 0:
                # Cluster centroids, filter them by extent and then sort them by ascending distance:
                db = DBSCAN(eps=0.3, min_samples=3).fit(filtered_pc)
                curr_segments, curr_centroids, curr_labels = dbscan_utils.collate_segmentation(db, 1.0)
                curr_centroids.sort(key=lambda x: np.linalg.norm(x))

                # Go through the DBSCAN centroids of the current frame:
                for centroid_airsim in curr_centroids:
                    centroid_eng, dump = spatial_utils.convert_eng_airsim(centroid_airsim, [0, 0, 0])
                    centroid_eng[0] -= execution_time * curr_vel * 2.0  # Compensate for sensor sync
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
                                if centroid_vehicle[1] > 0:  # Positive y means left side.
                                    cone_color = process_camera(lidar_to_left_cam,
                                                                centroid_lidar,
                                                                left_cam,
                                                                left_image,
                                                                curr_cone, idx, left_copy)
                                else:
                                    cone_color = process_camera(lidar_to_right_cam,
                                                                centroid_lidar,
                                                                right_cam,
                                                                right_image,
                                                                curr_cone, idx, right_copy)
                            break
                    # If no centroid is close enough to an existing one, create a new tracker instance:
                    if not centroid_exists:
                        new_centroid = tracker_utils.ConeTracker(centroid_global)
                        tracked_cones.append(new_centroid)

            desired_steer = pursuit_follower.calc_ref_steering(tracked_cones, map_to_vehicle)
            desired_steer /= pursuit_follower.max_steering  # Convert range to [-1, 1]
            desired_steer = np.clip(desired_steer, -0.2, 0.2)  # Saturate

            shmem_setpoint.buf[:8] = struct.pack('d', desired_steer)
            real_steer = struct.unpack('d', shmem_output.buf[:8])[0]

            car_controls.steering = real_steer
            client.setCarControls(car_controls)
            execution_time = time.perf_counter() - last_iteration
            # print(execution_time)

            if idx > decimation:
                cv2.imwrite(os.path.join(image_dest, 'left_' + str(save_idx) + '.png'), left_copy)
                cv2.imwrite(os.path.join(image_dest, 'right_' + str(save_idx) + '.png'), right_copy)
                save_idx += 1
                idx = 0

            idx += 1
        else:
            time.sleep(0.001)

    if save_data:
        tracked_objects = {'cones': tracked_cones, 'pursuit': pursuit_follower.pursuit_points}
        with open(os.path.join(data_dest, 'mapping_session.pickle'), 'wb') as pickle_file:
            pickle.dump(tracked_objects, pickle_file)
        print('pickle saved')

    return tracked_cones, pursuit_follower.pursuit_points


if __name__ == '__main__':
    airsim_client = airsim.CarClient()
    airsim_client.confirmConnection()
    airsim_client.enableApiControl(True)

    steering_procedure_manager = path_control.SteeringProcManager()
    dump = mapping_loop(airsim_client)

    # Done! stop vehicle:
    steering_procedure_manager.terminate_steering_procedure()
    vehicle_controls = airsim_client.getCarControls()
    vehicle_controls.throttle = 0.0
    vehicle_controls.brake = 1.0
    airsim_client.setCarControls(vehicle_controls)

