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


def aggregate_detections(airsim_client, iterations=2):
    pointcloud = np.array([])
    for curr_iter in range(iterations):
        lidarData = airsim_client.getLidarData()
        pointcloud = np.append(pointcloud, np.array(lidarData.point_cloud, dtype=np.dtype('f4')))
    return np.reshape(pointcloud, (int(pointcloud.shape[0] / 3), 3))


if __name__ == '__main__':

    # Airsim is stupid, always spawns at zero. Must compensate using "playerstart" in unreal:
    starting_x = 10.0
    starting_y = 20.0

    # Constant transform matrices:
    vehicle_to_lidar = spatial_utils.tf_matrix_from_airsim_pose([2, 0, -0.1], [0, 0, 0])
    vehicle_to_left_cam = spatial_utils.tf_matrix_from_airsim_pose([2, -0.5, -0.5], [-40.0, -10.0, 0])
    vehicle_to_right_cam = spatial_utils.tf_matrix_from_airsim_pose([2, 0.5, -0.5], [40.0, -10.0, 0])

    # connect to the AirSim simulator
    client = airsim.CarClient()
    client.confirmConnection()
    spatial_utils.set_airsim_pose(client, [0.0, 0.0], [90.0, 0, 0])
    time.sleep(1.0)

    # client.enableApiControl(True)
    # car_controls = airsim.CarControls()
    # car_controls.throttle = 0.1
    # client.setCarControls(car_controls)

    tracked_centroids = []
    start_time = time.time()
    idx = 0
    while time.time() - start_time < 30:
        # Constant throttle 0.1 (speed in the future)
        # trackers over centroids
        # separate into left/right
        # control law for steering
        # add color detection for "shufuni"
        vehicle_pose = client.simGetVehiclePose()
        map_to_vehicle = spatial_utils.tf_matrix_from_airsim_object(vehicle_pose)

        transform_mat = np.matmul(map_to_vehicle, vehicle_to_lidar)

        # DBSCAN filtering is done on the sensor-frame
        # #TODO change that? vehicle Z from airsim is odd. maybe transform only to vehicle frame. or cameras.
        point_cloud = aggregate_detections(client)  # Airsim's stupid lidar implementation
        # point_cloud[:, 2] *= -1  # Z is reversed in Airsim because of flying convention
        filtered_pc = dbscan_utils.filter_cloud(point_cloud, 5.0, 10.0, -0.5, 1.0)
        # after it works, dont forget to throw z away
        # Also, remove centroid labels

        if filtered_pc.size > 0:
            db = DBSCAN(eps=0.3, min_samples=3).fit(filtered_pc)
            curr_segments, curr_centroids = dbscan_utils.collate_segmentation(db, 0.5)
            print(idx)

            # for label in curr_centroids:
            #     centroid_local = np.append(curr_centroids[label], 1)
            #     centroid_global = np.matmul(transform_mat, centroid_local)
            #     print(centroid_local[0:2], centroid_global[0:2])

            centroid_exists = False
            for label in curr_centroids:  # For each DBSCAN centroid of this frame,
                centroid_local = np.append(curr_centroids[label], 1)
                centroid_global = np.matmul(transform_mat, centroid_local)[:3]
                for centroid in tracked_centroids:  # Compare them to all the known ones
                    if centroid.check_proximity(centroid_global):
                        centroid_exists = True
                        centroid.process_detection(centroid_global)

                # If it is not close enough to an existing point, create a new tracker instance:
                if not centroid_exists:
                    tracked_centroids.append(tracker.ObjectTracker(centroid_global))

            idx += 1
            time.sleep(0.1)

    with open('tracker_session.pickle', 'wb') as pickle_file:
        pickle.dump(tracked_centroids, pickle_file)
    print('saved pickle data')
