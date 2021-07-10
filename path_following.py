#!/usr/bin/env python
from spline_utils import PathSpline
import numpy as np
import airsim
from scipy.spatial.transform import Rotation as Rot
import time
import pickle
import multiprocessing
from spatial_utils import set_initial_pose
from high_level_control import PathFollower
from pidf_controller import PidfControl
from wheel_steer_emulator import WheelsPlant


if __name__ == '__main__':
    # Define spline
    x = np.array([10.00, 10.00, 10.00, 10.00, 10.00, 6.00, -6.00, -18.00, -23.00, -23.00, -17.00, 0.0, 8.00])
    y = np.array([20.00, 10.00, -10.00, -40.00, -60.00, -73.00, -78.00, -70.00, -38.00, 10.00, 30.00, 31.00, 27.00])
    my_spline = PathSpline(x, y)
    my_spline.generate_spline(0.1)
    follow_handler = PathFollower(my_spline)
    follow_handler.k_vel *= 2.0

    # Define speed controller:
    speed_controller = PidfControl(0.1)
    # speed_controller.set_pidf(0.275, 0.3, 0.0, 0.044)
    speed_controller.set_pidf(0.05, 0.0, 0.0, 0.044)
    speed_controller.set_extrema(0.01, 0.01)
    speed_controller.alpha = 0.01

    # steer_controller = PidfControl(0.01)
    # steer_controller.set_pidf(900.0, 0.0, 42.0, 0.0)
    # steer_controller.set_extrema(0.01, 1.0)
    # steer_controller.alpha = 0.1
    # steer_emulator = WheelsPlant(0.01)
    # steer_input = multiprocessing.Value('f', 0.0)
    # steer_output = multiprocessing.Value('f', 0.0)
    # is_active = multiprocessing.Value('B', int(1))
    # steering_thread = multiprocessing.Process(target=steer_emulator.async_steering,
    #                                           args=(steer_controller, steer_input, steer_output, is_active),
    #                                           daemon=True)
    # steering_thread.start()
    # time.sleep(2.0)  # New process takes a lot of time to "jumpstart"

    # Airsim is stupid, always spawns at zero. Must compensate using "playerstart" in unreal:
    starting_x = 10.0
    starting_y = 20.0

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

    start_time = time.time()
    while time.time() - start_time < 90.0:
        car_state = client.getCarState()
        car_pose = client.simGetVehiclePose()
        # kinematics estimated is the airsim dead reckoning!
        # curr_pos = [car_state.kinematics_estimated.position.x_val, car_state.kinematics_estimated.position.y_val]
        curr_pos = [car_pose.position.x_val + starting_x, car_pose.position.y_val + starting_y]
        curr_vel = car_state.speed
        orient = car_pose.orientation
        quat = np.array([orient.x_val, orient.y_val, orient.z_val, orient.w_val])
        rot = Rot.from_quat(quat)
        curr_heading = rot.as_euler('xyz', degrees=False)[2]
        # desired_speed, desired_steer = follow_handler.calc_ref_speed_steering(curr_pos, curr_vel, curr_heading)
        desired_speed, desired_steer, idx, distance, tangent, teta_e, teta_f = follow_handler.calc_ref_speed_steering(curr_pos, curr_vel, curr_heading)
        # desired_steer /= follow_handler.max_steering  # Convert range to [-1, 1]

        # Close a control loop over the throttle/speed of the vehicle:
        throttle_command = speed_controller.velocity_control(desired_speed, 0, curr_vel)

        # steer_input.value = desired_steer
        # desired_steer = steer_output.value / follow_handler.max_steering
        # print(steer_output.value, steer_output.value)

        car_controls.throttle = throttle_command
        car_controls.steering = -desired_steer
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

    with open('car_data.pickle', 'wb') as car_file:
        pickle.dump(car_data, car_file)
    print('saved car data')
    with open('control_data.pickle', 'wb') as control_file:
        pickle.dump(control_data, control_file)
    print('saved control data')
