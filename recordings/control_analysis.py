import numpy as np
import pickle
import csv
import os
import airsim
import pidf_controller
import spatial_utils
import discrete_plant_emulator
import time
from matplotlib import pyplot as plt

csv_folder = os.path.join(os.getcwd(), 'csv')
client = airsim.CarClient()
car_controls = airsim.CarControls()
step_data = np.ndarray(shape=(0, 3))
control_data = np.ndarray(shape=(0, 4))

bump_test = False
step_response = False

if bump_test:
    client.confirmConnection()
    client.enableApiControl(True)
    # Initialize vehicle starting point
    spatial_utils.set_airsim_pose(client, [40.0, -80.0], [90.0, 0, 0])
    time.sleep(1.0)

    low_step = 0.1
    high_step = 0.5
    car_controls.throttle = low_step
    client.setCarControls(car_controls)
    start_time1 = time.perf_counter()
    last_iteration = start_time1
    while time.perf_counter() - start_time1 < 6.0:
        if time.perf_counter() - last_iteration > 0.01:
            car_state = client.getCarState()
            step_data = np.append(step_data, [[time.perf_counter() - start_time1, low_step, car_state.speed]], axis=0)
            last_iteration = time.perf_counter()

    car_controls.throttle = high_step
    client.setCarControls(car_controls)
    start_time2 = time.perf_counter()
    last_iteration = start_time2
    while time.perf_counter() - start_time2 < 10.0:
        if time.perf_counter() - last_iteration > 0.01:
            car_state = client.getCarState()
            step_data = np.append(step_data, [[time.perf_counter() - start_time1, high_step, car_state.speed]], axis=0)
            last_iteration = time.perf_counter()

    car_controls.throttle = 0.0
    client.setCarControls(car_controls)

    with open(os.path.join(csv_folder, 'bump_test_' + str(high_step) + '.csv'), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['time', 'step_size', 'car_speed'])
        writer.writerows(step_data)

    fig, ax = plt.subplots(1, 1)
    ax.grid(True)
    ax.axis('equal')

    ax.plot(step_data[:, 0], step_data[:, 1], color='blue')
    ax.plot(step_data[:, 0], step_data[:, 2], color='red')

    fig.show()
    plt.waitforbuttonpress()

elif step_response:
    client.confirmConnection()
    client.enableApiControl(True)
    # Initialize vehicle starting point
    spatial_utils.set_airsim_pose(client, [40.0, -80.0], [90.0, 0, 0])
    time.sleep(1.0)

    # Define speed controller:
    speed_controller = pidf_controller.PidfControl(0.01)
    speed_controller.set_pidf(0.4, 0.4, 0.0, 0.0)
    speed_controller.set_extrema(min_setpoint=0.01, max_integral=1.0)
    # speed_controller.set_pidf(0.04, 0.04, 0.0, 0.043)
    # speed_controller.set_extrema(min_setpoint=0.01, max_integral=0.1)
    speed_controller.alpha = 0.01

    desired_speed = 7.0
    start_time = time.perf_counter()
    last_iteration = start_time
    while time.perf_counter() - start_time < 10.0:
        if time.perf_counter() - last_iteration > 0.01:
            car_state = client.getCarState()
            throttle = speed_controller.velocity_control(desired_speed, 0, car_state.speed)
            throttle = np.clip(throttle, 0.0, 1.0)
            car_controls.throttle = throttle
            client.setCarControls(car_controls)
            timestamp = time.perf_counter() - start_time
            control_data = np.append(control_data, [[timestamp, desired_speed, car_state.speed, throttle]], axis=0)
            last_iteration = time.perf_counter()

    car_controls.throttle = 0.0
    client.setCarControls(car_controls)

    with open(os.path.join(csv_folder, 'controlled_step_response.csv'), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['time', 'desired_speed', 'car_speed', 'throttle'])
        writer.writerows(control_data)

    fig, ax = plt.subplots(1, 1)
    ax.grid(True)
    ax.axis('equal')

    ax.plot(control_data[:, 0], control_data[:, 1], color='blue')
    ax.plot(control_data[:, 0], control_data[:, 2], color='red')
    ax.plot(control_data[:, 0], control_data[:, 3], color='green')

    fig.show()
    plt.waitforbuttonpress()

else:
    duration = 0.1
    dt = 0.001
    plant_data = np.ndarray(shape=(0,4), dtype=float)
    steer_emulator1 = discrete_plant_emulator.DiscretePlant(dt, 10, 4)
    steer_controller1 = pidf_controller.PidfControl(dt)
    steer_controller1.set_pidf(1000.0, 0.0, 12.6, 0.0)
    steer_controller1.set_extrema(0.01, 1.0)
    steer_controller1.alpha = 0.01

    steer_emulator2 = discrete_plant_emulator.DiscretePlant(dt, 10, 4)
    steer_controller2 = pidf_controller.PidfControl(dt)
    steer_controller2.set_pidf(1000.0, 0.0, 15.0, 0.0)
    steer_controller2.set_extrema(0.01, 1.0)
    steer_controller2.alpha = 0.01

    setpoint = 40.0
    output1 = 0.0
    output2 = 0.0
    idx = 0

    while idx * dt < duration:

        compensated_signal1 = steer_controller1.position_control(setpoint, output1)
        output1 = steer_emulator1.iterate_step(compensated_signal1)

        compensated_signal2 = steer_controller2.position_control(setpoint, output2)
        output2 = steer_emulator2.iterate_step(compensated_signal2)

        idx += 1
        plant_data = np.append(plant_data, [[idx * dt, setpoint, output1, output2]], axis=0)

    # Plot the result
    with open(os.path.join(csv_folder, 'compensated_discrete.csv'), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['time', 'setpoint', 'output1', 'output2'])
        writer.writerows(plant_data)
    print('saved csv data')
    timeline = np.linspace(0, duration, idx)
    fig, ax = plt.subplots(1, 1)
    ax.plot(timeline, plant_data[:, 1], '-k')
    ax.plot(timeline, plant_data[:, 2], '-b')
    ax.plot(timeline, plant_data[:, 3], '-r')
    ax.grid(True)
    fig.show()
    a = 5
    plt.waitforbuttonpress()


