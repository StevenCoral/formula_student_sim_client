from spline_utils import PathSpline
import numpy as np
import airsim
from scipy.spatial.transform import Rotation as Rot
import time
from matplotlib import pyplot as plt
from pidf_controller import PidfControl
import csv
import pickle


# def set_initial_pose(airsim_client, desired_position, desired_heading):
#     initial_pose = airsim_client.simGetVehiclePose()
#     rot = Rot.from_euler('xyz', [0, 0, desired_heading], degrees=True)
#     quat = rot.as_quat()
#     initial_pose.orientation.x_val = quat[0]
#     initial_pose.orientation.y_val = quat[1]
#     initial_pose.orientation.z_val = quat[2]
#     initial_pose.orientation.w_val = quat[3]
#     initial_pose.position.x_val = desired_position[0]
#     initial_pose.position.y_val = desired_position[1]
#     # initial_pose.position.z_val = desired_position[2]
#     airsim_client.simSetVehiclePose(initial_pose, ignore_collison=True)


def run():
    # Airsim is stupid, always spawns at zero. Must compensate using "playerstart" in unreal:
    speed_controller = PidfControl(0.1)
    # speed_controller.set_pidf(0.275, 0.3, 0.0, 0.044)
    speed_controller.set_pidf(0.1, 0.0, 0.0, 0.044)
    speed_controller.set_extrema(0.01, 0.01)
    speed_controller.alpha = 0.01
    inputs = np.array([], dtype=float)
    outputs = np.array([], dtype=float)

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

    try:
        duration = 20
        current_time = time.time()
        start_time = current_time
        last_iteration = current_time
        run_time = current_time - start_time
        idx = 0
        new_setpoint = 1.0

        while run_time <= duration:
            current_time = time.time()
            iteration_time = current_time - last_iteration
            run_time = current_time - start_time

            if iteration_time >= 0.1:
                if run_time > 5.0:
                    new_setpoint = 2.0
                if run_time > 10.0:
                    new_setpoint = 4.0
                if run_time > 15.0:
                    new_setpoint = 8.0

                car_state = client.getCarState()
                curr_vel = car_state.speed
                throttle_command = speed_controller.velocity_control(new_setpoint, 0, curr_vel)

                car_controls.throttle = throttle_command
                client.setCarControls(car_controls)
                idx += 1
                inputs = np.append(inputs, new_setpoint)
                outputs = np.append(outputs, curr_vel)
                last_iteration = time.time()

        car_controls.throttle = 0.0
        client.setCarControls(car_controls)

        # Plot the result
        timeline = np.linspace(0, duration, idx)
        fig, ax = plt.subplots(1, 1)
        ax.plot(timeline, inputs, '-r')
        ax.plot(timeline, outputs, '-b')
        # ax.plot(inputs, '-r')
        # ax.plot(outputs, '-b')
        ax.grid(True)
        fig.show()
        pass

    except KeyboardInterrupt:
        car_controls.throttle = 0.0
        client.setCarControls(car_controls)
        time.sleep(1.0)

    finally:
        car_controls.throttle = 0.0
        client.setCarControls(car_controls)
        time.sleep(1.0)


if __name__ == '__main__':
    run()
