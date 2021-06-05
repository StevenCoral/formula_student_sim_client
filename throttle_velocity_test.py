#!/usr/bin/env python
from spline_utils import PathSpline
import numpy as np
import airsim
from scipy.spatial.transform import Rotation as Rot
import time
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
    car_data = np.ndarray(shape=(0, 4))
    control_data = np.ndarray(shape=(0, 8))

    target_throttle = 0.9
    car_controls.throttle = target_throttle
    client.setCarControls(car_controls)

    start_time = time.time()
    while time.time() - start_time < 20.0:
        car_state = client.getCarState()
        car_pose = client.simGetVehiclePose()
        curr_pos = [car_pose.position.x_val + starting_x, car_pose.position.y_val + starting_y]
        curr_vel = car_state.speed
        orient = car_pose.orientation
        quat = np.array([orient.x_val, orient.y_val, orient.z_val, orient.w_val])
        rot = Rot.from_quat(quat)
        curr_heading = rot.as_euler('xyz', degrees=False)[2]



        new_car_data = [car_state.kinematics_estimated.position.x_val,
                        car_state.kinematics_estimated.position.y_val,
                        curr_heading,
                        curr_vel]
        car_data = np.append(car_data, [new_car_data], axis=0)

        print('current timestamp: ', time.time() - start_time)
        time.sleep(0.1)

    print('average velocity for throttle:', target_throttle, 'is:', np.average(car_data[50:150, 3]), 'm/s')

