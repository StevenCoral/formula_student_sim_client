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
    car_data = np.ndarray(shape=(0, 2))
    control_data = np.ndarray(shape=(0, 8))

    target_throttle = 0.5
    car_controls.throttle = target_throttle
    client.setCarControls(car_controls)

    start_time = time.time()
    while time.time() - start_time < 12.0:
        time_diff = time.time() - start_time
        car_state = client.getCarState()
        car_pose = client.simGetVehiclePose()
        curr_pos = [car_pose.position.x_val + starting_x, car_pose.position.y_val + starting_y]
        curr_vel = car_state.speed

        new_car_data = [time_diff, curr_vel]
        car_data = np.append(car_data, [new_car_data], axis=0)

        # print('current timestamp: ', time_diff)
        time.sleep(0.01)

    car_controls.throttle = 0.0
    client.setCarControls(car_controls)

    # Begin data processing:
    with open('bump_test_torque_2000_step_' + str(target_throttle) + '.csv', 'w', newline='', encoding='utf-8') as f:
        # create the csv writer
        writer = csv.writer(f)
        writer.writerow(['time [s]', 'speed [m/s]'])
        writer.writerows(car_data)

    total_len = car_data.shape[0]
    print('average velocity for throttle:', target_throttle, 'is:', np.average(car_data[2*total_len//3:, 1]), 'm/s')
    fig, ax = plt.subplots(1, 1)
    ax.plot(car_data[:, 0], car_data[:, 1], '.b')
    fig.show()
    pass

