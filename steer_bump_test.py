#!/usr/bin/env python
from spline_utils import PathSpline
import numpy as np
import airsim
from scipy.spatial.transform import Rotation as Rot
import time
from matplotlib import pyplot as plt
import csv
import pickle


def set_initial_pose(airsim_client, desired_position, desired_heading):
    initial_pose = airsim_client.simGetVehiclePose()
    rot = Rot.from_euler('xyz', [0, 0, desired_heading], degrees=True)
    quat = rot.as_quat()
    initial_pose.orientation.x_val = quat[0]
    initial_pose.orientation.y_val = quat[1]
    initial_pose.orientation.z_val = quat[2]
    initial_pose.orientation.w_val = quat[3]
    initial_pose.position.x_val = desired_position[0]
    initial_pose.position.y_val = desired_position[1]
    # initial_pose.position.z_val = desired_position[2]
    airsim_client.simSetVehiclePose(initial_pose, ignore_collison=True)


if __name__ == '__main__':

    # Airsim is stupid, always spawns at zero. Must compensate using "playerstart" in unreal:
    starting_x = 15.0
    starting_y = 100.0

    # connect to the AirSim simulator
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    set_initial_pose(client, [0.0, 0.0], -90.0)
    time.sleep(1.0)
    # client.enableApiControl(False)

    car_controls = airsim.CarControls()
    car_data = np.ndarray(shape=(0, 3))

    steer_setpoint = 1.0

    start_time = time.time()
    while time.time() - start_time < 10.0:
        time_diff = time.time() - start_time
        if time_diff > 3.0:
            steer_setpoint = -1.0
        if time_diff > 6.0:
            steer_setpoint = 1.0
        if time_diff > 9.0:
            steer_setpoint = -1.0

        car_controls.steering = steer_setpoint
        client.setCarControls(car_controls)

        car_controls2 = client.getCarControls()
        curr_steer = car_controls2.steering

        new_car_data = [time_diff, steer_setpoint, curr_steer]
        car_data = np.append(car_data, [new_car_data], axis=0)

        time.sleep(0.01)

    car_controls.throttle = 0.0
    client.setCarControls(car_controls)

    total_len = car_data.shape[0]
    fig, ax = plt.subplots(1, 1)
    ax.plot(car_data[:, 0], car_data[:, 1], '-b')
    ax.plot(car_data[:, 0], car_data[:, 2], '-r')
    fig.show()
    pass

