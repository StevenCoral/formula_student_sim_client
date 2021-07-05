#!/usr/bin/env python
from spline_utils import PathSpline
import numpy as np
from matplotlib import pyplot as plt
from pidf_controller import PidfControl
from scipy.spatial.transform import Rotation as Rot
import time
import pickle


class WheelsPlant:
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
        y = self.u0 * u + self.u1 * self.u_k1 + self.u2 * self.u_k2 + self.y1 * self.y_k1 + self.y2 * self.y_k2
        # Push values of u and y one step backwards:
        self.u_k2 = self.u_k1
        self.u_k1 = u
        self.y_k2 = self.y_k1
        self.y_k1 = y

        return y


if __name__ == '__main__':
    duration = 3
    timeline = np.linspace(0, duration, duration*100)
    inputs = np.array([], dtype=float)
    outputs = np.array([], dtype=float)
    my_plant = WheelsPlant(0.01)
    compensator = PidfControl(0.01)
    compensator.set_pidf(200, 0, 20, 0)
    compensator.set_extrema(0.01, 1)
    compensator.alpha = 0.1
    pass
    y = 0
    for idx in range(len(timeline)):
        if idx > 200:
            a=1
        u = compensator.position_control(1.0, y)
        y = my_plant.iterate_step(u)

        inputs = np.append(inputs, u)
        outputs = np.append(outputs, y)

    # plot the result
    fig, ax = plt.subplots(1, 1)
    # ax.plot(timeline, inputs, '-r')
    ax.plot(timeline, outputs, '-b')
    ax.grid(True)
    fig.show()
    pass
