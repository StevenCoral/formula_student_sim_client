#!/usr/bin/env python
from spline_utils import PathSpline
import numpy as np
import airsim
from scipy.spatial.transform import Rotation as Rot
import time
import pickle
import multiprocessing
import spatial_utils
from path_control import PathFollower
from pidf_controller import PidfControl
from wheel_steer_emulator import WheelsPlant


def following_loop(client, spline_obj=None):

    save_data = False

    if spline_obj is None:
        # Airsim is stupid, always spawns at zero. Must compensate using "playerstart" location in unreal:
        starting_x = 10.0
        starting_y = 20.0

        # Define spline
        x = np.array([10.00, 10.00, 10.00, 10.00, 10.00, 6.00, -6.00, -18.00, -23.00, -23.00, -17.00, 0.0, 8.00])
        y = np.array([20.00, 10.00, -10.00, -40.00, -60.00, -73.00, -78.00, -70.00, -38.00, 10.00, 30.00, 31.00, 27.00])
        x -= starting_x
        y -= starting_y
        y *= -1

        spline_obj = PathSpline(x, y)
        spline_obj.generate_spline(0.1)
        spatial_utils.set_airsim_pose(client, [0.0, 0.0], [90.0, 0, 0])

    follow_handler = PathFollower(spline_obj)
    follow_handler.k_vel *= 2.0

    # Define speed controller:
    speed_controller = PidfControl(0.1)
    # speed_controller.set_pidf(0.275, 0.3, 0.0, 0.044)
    speed_controller.set_pidf(0.05, 0.0, 0.0, 0.044)
    speed_controller.set_extrema(0.01, 0.01)
    speed_controller.alpha = 0.01

    steer_controller = PidfControl(0.01)
    steer_controller.set_pidf(900.0, 0.0, 42.0, 0.0)
    steer_controller.set_extrema(0.01, 1.0)
    steer_controller.alpha = 0.01
    steer_emulator = WheelsPlant(0.01)
    # steer_input = multiprocessing.Value('f', 0.0)
    # steer_output = multiprocessing.Value('f', 0.0)
    # is_active = multiprocessing.Value('B', int(1))
    # steering_thread = multiprocessing.Process(target=steer_emulator.async_steering,
    #                                           args=(steer_controller, steer_input, steer_output, is_active),
    #                                           daemon=True)
    # steering_thread.start()
    # time.sleep(2.0)  # New process takes a lot of time to "jumpstart"

    car_controls = airsim.CarControls()
    car_data = np.ndarray(shape=(0, 4))
    control_data = np.ndarray(shape=(0, 8))

    start_time = time.time()
    while time.time() - start_time < 60.0:
        car_state = client.getCarState()
        car_pose = client.simGetVehiclePose()
        # kinematics estimated is the airsim dead reckoning!
        # curr_pos = [car_state.kinematics_estimated.position.x_val, car_state.kinematics_estimated.position.y_val]

        curr_vel = car_state.speed
        curr_pos, curr_rot = spatial_utils.extract_pose_from_airsim(car_pose)
        curr_heading = np.deg2rad(curr_rot[0])

        desired_speed, desired_steer, idx, distance, tangent, teta_e, teta_f = follow_handler.calc_ref_speed_steering(
            curr_pos, curr_vel, curr_heading)

        # Close a control loop over the throttle/speed of the vehicle:
        throttle_command = speed_controller.velocity_control(desired_speed, 0, curr_vel)

        # Close a control loop over the steering angle.
        # TODO can be done without multiprocessing, by increasing the rate of the control loop.
        # steer_input.value = desired_steer
        # steer_command = steer_output.value
        # steer_command /= follow_handler.max_steering  # Convert range to [-1, 1]
        desired_steer /= follow_handler.max_steering  # Convert range to [-1, 1]

        car_controls.throttle = throttle_command
        car_controls.steering = desired_steer
        client.setCarControls(car_controls)

        new_car_data = [car_state.kinematics_estimated.position.x_val,
                        car_state.kinematics_estimated.position.y_val,
                        curr_heading,
                        curr_vel]
        car_data = np.append(car_data, [new_car_data], axis=0)

        new_control_data = [desired_speed,
                            desired_steer,
                            idx,
                            distance,
                            tangent[0],
                            tangent[1],
                            teta_e,
                            teta_f]
        control_data = np.append(control_data, [new_control_data], axis=0)

        print('current timestamp: ', time.time() - start_time)
        time.sleep(0.1)

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
