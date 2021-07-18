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
import tracker
import os


def aggregate_detections(airsim_client, iterations=2):
    pointcloud = np.array([])
    for curr_iter in range(iterations):
        lidarData = airsim_client.getLidarData()
        pointcloud = np.append(pointcloud, np.array(lidarData.point_cloud, dtype=np.dtype('f4')))
    return np.reshape(pointcloud, (int(pointcloud.shape[0] / 3), 3))


def compare_trackers(tracker_list):
    centroid_exists = False
    # Compare them against all the known tracked objects:
    for centroid in tracker_list:
        if centroid.check_proximity(centroid_global):
            centroid_exists = True
            centroid.process_detection(centroid_global)

    # If it is not close enough to an existing point, create a new tracker instance:
    if not centroid_exists:
        tracker_list.append(tracker.ObjectTracker(centroid_global))


def save_img(airsim_response, file_path):
    # get numpy array
    img1d = np.fromstring(airsim_response.image_data_uint8, dtype=np.uint8)
    # reshape array to 4 channel image array H X W X 4
    img_rgb = img1d.reshape(airsim_response.height, airsim_response.width, 3)
    # original image is fliped vertically
    img_rgb = np.flipud(img_rgb)
    # write to png
    airsim.write_png(file_path, img_rgb)


if __name__ == '__main__':

    # Airsim is stupid, always spawns at zero. Must compensate using "playerstart" in unreal:
    starting_x = 10.0
    starting_y = 20.0
    image_dest = 'D:\MscProject\BGR_client\images'

    # Constant transform matrices:
    # Notating A_to_B means that later taking a vectir in frame A and multiplying in the matrix
    # will result in the same world point in frame B, even though the definition of the deltas
    # within the parentheses describe the transformation from B to A.

    lidar_to_vehicle = spatial_utils.tf_matrix_from_airsim_pose([2, 0, -0.1], [0, 0, 0])
    left_cam_to_vehicle = spatial_utils.tf_matrix_from_airsim_pose([2, -0.5, -0.5], [-40.0, -10.0, 0])
    right_cam_to_vehicle = spatial_utils.tf_matrix_from_airsim_pose([2, 0.5, -0.5], [40.0, -10.0, 0])
    lidar_to_left_cam = np.matmul(np.linalg.inv(left_cam_to_vehicle), lidar_to_vehicle)
    lidar_to_right_cam = np.matmul(np.linalg.inv(right_cam_to_vehicle), lidar_to_vehicle)

    cam_width = 640
    cam_height = 360
    cam_FOV = 90.0
    cam_focal = 0.5 * cam_width / (np.tan(np.deg2rad(cam_FOV) / 2))
    cam_intrinsics = np.zeros(shape=(3, 3))
    cam_intrinsics[0, 0] = cam_focal
    cam_intrinsics[1, 1] = cam_focal
    cam_intrinsics[2, 2] = 1.0
    cam_intrinsics[0, 2] = cam_width / 2
    cam_intrinsics[1, 2] = cam_height / 2

    # connect to the AirSim simulator
    client = airsim.CarClient()
    client.confirmConnection()
    spatial_utils.set_airsim_pose(client, [0.0, 0.0], [90.0, 0, 0])
    time.sleep(1.0)

    # client.enableApiControl(True)
    # car_controls = airsim.CarControls()
    # car_controls.throttle = 0.1
    # client.setCarControls(car_controls)

    right_centroids = []
    left_centroids = []
    start_time = time.time()
    idx = 0
    while time.time() - start_time < 40:
        # Constant throttle 0.1 (speed in the future)
        # trackers over centroids
        # separate into left/right
        # control law for steering
        # add color detection for "shufuni"
        vehicle_pose = client.simGetVehiclePose()
        vehicle_to_map = spatial_utils.tf_matrix_from_airsim_object(vehicle_pose)

        lidar_to_map = np.matmul(vehicle_to_map, lidar_to_vehicle)

        # DBSCAN filtering is done on the sensor-frame
        point_cloud = aggregate_detections(client)  # Airsim's stupid lidar implementation requires aggregation
        filtered_pc = dbscan_utils.filter_cloud(point_cloud, 3.0, 15.0, -0.5, 1.0)

        print(idx)
        if filtered_pc.size > 0:
            db = DBSCAN(eps=0.3, min_samples=3).fit(filtered_pc)
            curr_segments, curr_centroids, curr_labels = dbscan_utils.collate_segmentation(db, 1.0)

            # responses = client.simGetImages([airsim.ImageRequest("LeftCam", airsim.ImageType.Scene, False, False),
            #                                  airsim.ImageRequest("RightCam", airsim.ImageType.Scene, False, False)])
            # save_img(responses[0], os.path.join(image_dest, 'left_' + str(idx)))
            # save_img(responses[1], os.path.join(image_dest, 'right_' + str(idx)))

            # Go through the DBSCAN centroids of the current frame:
            for centroid_airsim in curr_centroids:
                # centroid_exists = False
                centroid_eng, dump = spatial_utils.convert_eng_airsim(centroid_airsim, [0, 0, 0])
                centroid_local = np.append(centroid_eng, 1)
                centroid_global = np.matmul(lidar_to_map, centroid_local)[:3]
                centroid_vehicle = np.matmul(lidar_to_vehicle, centroid_local)[:3]

                if centroid_vehicle[1] > 0:  # Positive y means left side.
                    compare_trackers(left_centroids)
                else:
                    compare_trackers(right_centroids)
                #     # Compare them against all the known tracked objects:
                #     for centroid in left_centroids:
                #         if centroid.check_proximity(centroid_global):
                #             centroid_exists = True
                #             centroid.process_detection(centroid_global)
                #
                #     # If it is not close enough to an existing point, create a new tracker instance:
                #     if not centroid_exists:
                #         left_centroids.append(tracker.ObjectTracker(centroid_global))
                # else:
                #     # Compare them against all the known tracked objects:
                #     for centroid in right_centroids:
                #         if centroid.check_proximity(centroid_global):
                #             centroid_exists = True
                #             centroid.process_detection(centroid_global)
                #
                #     # If it is not close enough to an existing point, create a new tracker instance:
                #     if not centroid_exists:
                #         right_centroids.append(tracker.ObjectTracker(centroid_global))

            idx += 1
            time.sleep(0.1)

    tracked_centroids = {'left': left_centroids, 'right': right_centroids}
    with open('tracker_session.pickle', 'wb') as pickle_file:
        pickle.dump(tracked_centroids, pickle_file)
    print('saved pickle data')
