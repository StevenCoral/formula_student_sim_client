
from spline_utils import PathSpline
import numpy as np
import airsim
from matplotlib import pyplot as plt
from pidf_controller import PidfControl
from wheel_steer_emulator import WheelsPlant
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
    duration = 3
    timeline = np.linspace(0, duration, duration*100)
    inputs = np.array([], dtype=float)
    outputs = np.array([], dtype=float)
    wheels_plant = WheelsPlant(0.01)
    compensator = PidfControl(0.01)
    compensator.set_pidf(900, 0, 42, 0)
    compensator.set_extrema(0.01, 1)
    compensator.alpha = 0.5
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
        desired_speed, desired_steer, idx, distance, tangent, teta_e, teta_f = follow_handler.calc_ref_speed_steering(
            curr_pos, curr_vel, curr_heading)
        desired_steer /= follow_handler.max_steering

        # car_controls.throttle = 0.6
        car_controls.throttle = desired_speed / 16.0  # Something like...
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
