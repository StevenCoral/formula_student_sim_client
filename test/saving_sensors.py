#!/usr/bin/env python
from spline_utils import PathSpline
import numpy as np
import airsim
from scipy.spatial.transform import Rotation as Rot
import time
import pickle
import multiprocessing
from spatial_utils import set_initial_pose
import csv


if __name__ == '__main__':

    # Airsim is stupid, always spawns at zero. Must compensate using "playerstart" in unreal:
    starting_x = 10.0
    starting_y = 20.0

    # connect to the AirSim simulator
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    set_initial_pose(client, [0.0, 0.0], -90.0)
    time.sleep(2.0)

    lidarData = client.getLidarData()
    point_cloud = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
    point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 3), 3))

    with open('pointcloud.pickle', 'wb') as pickle_file:
        pickle.dump(point_cloud, pickle_file)
    print('saved pickle data')

    with open('pointcloud.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['x', 'y', 'z'])
        writer.writerows(point_cloud)
    print('saved csv data')
