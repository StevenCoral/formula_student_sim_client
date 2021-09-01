
from spline_utils import PathSpline
import numpy as np
from matplotlib import pyplot as plt
from pidf_controller import PidfControl
from scipy.spatial.transform import Rotation as Rot
import time
import pickle
import csv
import threading
import multiprocessing
import airsim


class DiscretePlant:
    def __init__(self, ts, k=1, a=1):
        # Emulates a discrete plant of the form: k/(s^2+as).
        alpha = 1/(2*a*ts + 4)
        # Calculating u and y coefficients:
        self.u0 = alpha*k*ts**2
        self.u1 = 2*alpha*k*ts**2
        self.u2 = alpha*k*ts**2
        self.y1 = alpha*8
        self.y2 = alpha*(2*a*ts-4)

        # Saving previous u and y values:
        self.u_k1 = 0  # Meaning U_k-1
        self.u_k2 = 0
        self.y_k1 = 0
        self.y_k2 = 0

    def iterate_step(self, u):
        # Calculate the next value:
        y = self.u0 * u + self.u1 * self.u_k1 + self.u2 * self.u_k2 + self.y1 * self.y_k1 + self.y2 * self.y_k2
        # Push values of u and y one step backwards:
        self.u_k2 = self.u_k1
        self.u_k1 = u
        self.y_k2 = self.y_k1
        self.y_k1 = y
        return y

    def async_steering(self, controller, setpoint, output, is_active):
        while is_active:
            compensated_signal = controller.position_control(setpoint.value, output.value)
            output.value = self.iterate_step(compensated_signal)
            time.sleep(0.009)


def run_subprocess():
    duration = 4
    inputs = np.array([], dtype=float)
    outputs = np.array([], dtype=float)
    steer_emulator = DiscretePlant(0.01)
    steer_controller = PidfControl(0.01)
    steer_controller.set_pidf(900.0, 0.0, 42.0, 0.0)
    steer_controller.set_extrema(0.01, 1.0)
    steer_controller.alpha = 0.1

    # connect to the AirSim simulator
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    # client.enableApiControl(False)
    time.sleep(1.0)
    car_controls = airsim.CarControls()

    setpoint = multiprocessing.Value('f', 10.0)
    output = multiprocessing.Value('f', 0.0)
    is_active = multiprocessing.Value('B', int(1))
    steering_thread = multiprocessing.Process(target=steer_emulator.async_steering,
                                              args=(steer_controller, setpoint, output, is_active),
                                              daemon=True)
    steering_thread.start()
    time.sleep(2.0)  # New process takes a lot of time to "jumpstart"

    try:
        current_time = time.time()
        start_time = current_time
        last_iteration = current_time
        run_time = current_time - start_time
        iteration_time = current_time - last_iteration
        idx = 0
        new_setpoint = 10.0
        while run_time <= duration:
            current_time = time.time()
            iteration_time = current_time - last_iteration
            run_time = current_time - start_time

            if iteration_time >= 0.1:
                if run_time > 1.0:
                    new_setpoint = -20.0
                if run_time > 2.0:
                    new_setpoint = 30.0
                if run_time > 3.0:
                    new_setpoint = -40.0

                car_controls.steering = new_setpoint
                client.setCarControls(car_controls)

                setpoint.value = new_setpoint
                idx += 1
                inputs = np.append(inputs, setpoint.value)
                outputs = np.append(outputs, output.value)
                last_iteration = time.time()

        is_active.value = int(0)  # Stop calculating angle
        # Plot the result
        timeline = np.linspace(0, duration, idx)
        fig, ax = plt.subplots(1, 1)
        ax.plot(timeline, inputs, '-r')
        ax.plot(timeline, outputs, '-b')
        ax.grid(True)
        fig.show()
        pass

    except KeyboardInterrupt:
        is_active.value = int(0)  # Stop calculating angle
        time.sleep(1.0)

    finally:
        is_active.value = int(0)  # Stop calculating angle
        time.sleep(1.0)


def run_offline():
    duration = 4
    inputs = np.array([], dtype=float)
    outputs = np.array([], dtype=float)
    steer_emulator = DiscretePlant(0.001, 10, 4)
    steer_controller = PidfControl(0.001)
    # steer_controller.set_pidf(900.0, 0.0, 42.0, 0.0)
    steer_controller.set_pidf(1000.0, 0.0, 15, 0.0)
    steer_controller.set_extrema(0.01, 1.0)
    steer_controller.alpha = 0.01

    setpoint = 10.0
    output = 0.0
    idx = 0
    current_time = time.time()
    start_time = current_time
    last_iteration = current_time
    run_time = current_time - start_time
    iteration_time = current_time - last_iteration

    while idx < duration * 1000:
        current_time = time.time()
        iteration_time = current_time - last_iteration
        run_time = current_time - start_time

        if idx > 1.0 * 1000:
            setpoint = -20.0
        if idx > 2.0 * 1000:
            setpoint = 30.0
        if idx > 3.0 * 1000:
            setpoint = -40.0

        compensated_signal = steer_controller.position_control(setpoint, output)
        output = steer_emulator.iterate_step(compensated_signal)

        idx += 1
        inputs = np.append(inputs, setpoint)
        outputs = np.append(outputs, output)
        last_iteration = time.time()

    # Plot the result
    # save_data = np.append(inputs.reshape((inputs.size, 1)), outputs.reshape((outputs.size, 1)), axis=1)
    # #{'inputs': inputs, 'outputs': outputs}
    # with open('discrete.csv', 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow(['inputs', 'outputs'])
    #     writer.writerows(save_data)
    # print('saved csv data')
    timeline = np.linspace(0, duration, idx)
    fig, ax = plt.subplots(1, 1)
    ax.plot(timeline, inputs, '-r')
    ax.plot(timeline, outputs, '-b')
    ax.grid(True)
    fig.show()
    pass


if __name__ == '__main__':
    # run_subprocess()
    run_offline()
