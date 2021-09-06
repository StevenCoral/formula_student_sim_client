#!/usr/bin/env python
from spline_utils import PathSpline
import numpy as np
import airsim
import time
import pickle
import spatial_utils
import path_control
from pidf_controller import PidfControl
import struct
import csv
import os


def following_loop(client, spline_obj=None):
    data_dest = os.path.join(os.getcwd(), 'recordings')
    save_data = True

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

    # Define Stanley-method parameters:
    follow_handler = path_control.StanleyFollower(spline_obj)
    # follow_handler.k_vel *= 3.0
    follow_handler.max_velocity = 15.0  # m/s
    follow_handler.min_velocity = 10.0  # m/s
    follow_handler.lookahead = 5.0  # meters
    follow_handler.k_steer = 2.0  # Stanley steering coefficient

    # Open access to shared memory blocks:
    inputs = np.array([], dtype=float)
    outputs = np.array([], dtype=float)
    shmem_active, shmem_setpoint, shmem_output = path_control.SteeringProcManager.retrieve_shared_memories()

    # Define speed controller:
    speed_controller = PidfControl(0.01)
    speed_controller.set_pidf(0.05, 0.0, 0.0, 0.044)
    speed_controller.set_extrema(0.01, 0.01)
    speed_controller.alpha = 0.01

    # Initialize loop variables:
    loop_trigger = False
    leaving_distance = 10.0
    entering_distance = 4.0

    car_controls = airsim.CarControls()
    car_data = np.ndarray(shape=(0, 8))

    start_time = time.perf_counter()
    last_iteration = start_time
    sample_time = 0.01

    while last_iteration - start_time < 300: #TODO change back to 300
        now = time.perf_counter()
        delta_time = now - last_iteration

        if delta_time > sample_time:
            last_iteration = time.perf_counter()
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
            throttle_command = np.clip(throttle_command, 0.0, 1.0)

            desired_steer /= follow_handler.max_steering  # Convert range to [-1, 1]
            desired_steer = np.clip(desired_steer, -0.3, 0.3)  # Saturate

            shmem_setpoint.buf[:8] = struct.pack('d', desired_steer)
            real_steer = struct.unpack('d', shmem_output.buf[:8])[0]
            # inputs = np.append(inputs, desired_steer)
            # outputs = np.append(outputs, real_steer)

            car_controls.throttle = throttle_command
            car_controls.steering = real_steer # desired_steer
            client.setCarControls(car_controls)

            car_data = np.append(car_data,
                                 [[curr_pos[0], curr_pos[1], curr_rot[0],
                                   desired_speed, car_state.speed,
                                   desired_steer, real_steer, throttle_command]],
                                 axis=0)

    pickling_objects = {'path': follow_handler.path, 'car_data': car_data}
    if save_data:
        with open(os.path.join(data_dest, 'following_session.pickle'), 'wb') as pickle_file:
            pickle.dump(pickling_objects, pickle_file)
        print('saved pickle data')
        with open('car_data.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['x', 'y', 'heading', 'v_desired', 'v_delivered',
                             's_desired', 's_delivered', 'throttle'])
            writer.writerows(car_data)
        print('saved csv data')
    return pickling_objects, car_data


if __name__ == '__main__':
    airsim_client = airsim.CarClient()
    airsim_client.confirmConnection()
    airsim_client.enableApiControl(True)

    steering_procedure_manager = path_control.SteeringProcManager()
    following_loop(airsim_client)

    # Done! stop vehicle:
    steering_procedure_manager.terminate_steering_procedure()
    vehicle_controls = airsim_client.getCarControls()
    vehicle_controls.throttle = 0.0
    vehicle_controls.brake = 1.0
    airsim_client.setCarControls(vehicle_controls)
