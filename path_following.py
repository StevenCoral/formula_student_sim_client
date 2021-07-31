#!/usr/bin/env python
from spline_utils import PathSpline
import numpy as np
import airsim
from scipy.spatial.transform import Rotation as Rot
import time
import pickle
import multiprocessing
import spatial_utils
from path_control import StanleyFollower
from pidf_controller import PidfControl
from wheel_steer_emulator import WheelsPlant


def following_loop(client, spline_obj=None):

    save_data = False

    if spline_obj is None:
        # Airsim is stupid, always spawns at zero. Must compensate using "playerstart" location in unreal:
        starting_x = 12
        starting_y = 19
        spline_origin_x = 12
        spline_origin_y = 25
        x_offset = spline_origin_x - starting_x
        y_offset = spline_origin_y - starting_y

        # Define spline
        x = np.array(
            [-2.00, 0.00, 0.00, 0.00, -5.00, -13.0, -21.0, -27.0, -32.0, -38.0, -47.0, -55.0, -53.0, -40.0, -25.0,
             -23.0, -37.0, -34.0, -20.0, -8.0])
        y = np.array(
            [6.0, -7.0, -19.0, -34.0, -46.0, -51.0, -54.0, -59.0, -68.0, -74.0, -75.0, -68.0, -54.0, -39.0, -23.0,
             -8.00, 6.00, 21.00, 23.00, 15.00])
        x += x_offset
        y += y_offset
        y *= -1
        spline_obj = PathSpline(x, y)
        spline_obj.generate_spline(0.1, smoothing=1)
        spatial_utils.set_airsim_pose(client, [0.0, 0.0], [90.0, 0, 0])

    follow_handler = StanleyFollower(spline_obj)
    follow_handler.k_vel *= 3.0
    follow_handler.max_velocity = 15.0  # m/s
    follow_handler.min_velocity = 10.0  # m/s
    follow_handler.lookahead = 6.0  # meters
    follow_handler.k_steer = 2.0  # Stanley steering coefficient

    # Define speed controller:
    speed_controller = PidfControl(0.1)
    speed_controller.set_pidf(0.05, 0.0, 0.0, 0.044)
    speed_controller.set_extrema(0.01, 0.01)
    speed_controller.alpha = 0.01

    loop_trigger = False
    leaving_distance = 10.0
    entering_distance = 4.0

    car_controls = airsim.CarControls()
    car_data = np.ndarray(shape=(0, 4))
    control_data = np.ndarray(shape=(0, 8))

    start_time = time.time()
    last_iteration = start_time
    sample_time = 0.1

    while last_iteration - start_time < 300:
        delta_time = time.time() - last_iteration
        if delta_time > sample_time:
            last_iteration = time.time()
            vehicle_pose = client.simGetVehiclePose()
            vehicle_to_map = spatial_utils.tf_matrix_from_airsim_object(vehicle_pose)
            car_state = client.getCarState()
            curr_vel = car_state.speed
            curr_pos, curr_rot = spatial_utils.extract_pose_from_airsim(vehicle_pose)
            curr_heading = np.deg2rad(curr_rot[0])

            distance_from_start = np.linalg.norm(vehicle_to_map[0:2, 3])
            if not loop_trigger:
                if distance_from_start > leaving_distance:
                    loop_trigger = True
            else:
                if distance_from_start < entering_distance:
                    break

            desired_speed, desired_steer = follow_handler.calc_ref_speed_steering(curr_pos, curr_vel, curr_heading)

            # Close a control loop over the throttle/speed of the vehicle:
            throttle_command = speed_controller.velocity_control(desired_speed, 0, curr_vel)

            desired_steer /= follow_handler.max_steering  # Convert range to [-1, 1]
            desired_steer = np.clip(desired_steer, -0.3, 0.3)  # Saturate

            car_controls.throttle = throttle_command
            car_controls.steering = desired_steer
            client.setCarControls(car_controls)

        else:
            time.sleep(0.005)

    if save_data:
        with open('car_data.pickle', 'wb') as car_file:
            pickle.dump(car_data, car_file)
        print('saved car data')
        with open('control_data.pickle', 'wb') as control_file:
            pickle.dump(control_data, control_file)
        print('saved control data')


if __name__ == '__main__':
    airsim_client = airsim.CarClient()
    airsim_client.confirmConnection()
    airsim_client.enableApiControl(True)
    following_loop(airsim_client)

    # Done! stop vehicle:
    car_controls = airsim_client.getCarControls()
    car_controls.throttle = 0.0
    car_controls.brake = 1.0
    airsim_client.setCarControls(car_controls)
