#!/usr/bin/env python
from math import *
import numpy as np
import airsim


class PathFollower:
    def __init__(self, path_spline):
        self.path = path_spline

        # The rest of the parameters should be set post-initialization,
        # since it is usually less frequently changed (vehicle params).
        self.epsilon = 1e-6  # For numerical stability

        self.range_tol = path_spline.meters_per_index / 2.0  # meters
        self.max_velocity = 2.0  # m/s
        self.min_velocity = 1.0  # m/s
        self.max_steering = np.deg2rad(45.0)

        self.k_vel = (self.max_velocity - self.min_velocity) / (self.path.curvature.max() + self.epsilon)
        self.k_steer = 1.0  # Control coefficient
        self.lookahead = 1.0  # meters
        # self.lookahead_steer = 20  # meters

    def calc_ref_speed_steering(self, car_pos, car_vel, heading):
        # First we handle the desired steering angle [rad]:
        closest_idx, closest_dist, closest_tangent = self.path.find_closest_point(car_pos)
        theta_e = heading - np.arctan2(closest_tangent[1], closest_tangent[0])
        theta_f = np.arctan(self.k_steer * closest_dist / (np.abs(car_vel) + self.epsilon))
        steering_angle = np.clip(theta_e + theta_f, -self.max_steering, self.max_steering)

        # Then we handle the desired speed [m/s]:
        lookahead_idx = closest_idx + int(np.around(self.lookahead / (self.path.meters_per_index + self.epsilon)))
        speed = self.max_velocity - self.k_vel * self.path.curvature[lookahead_idx]

        return speed, steering_angle
        # get airsim pose
        # extract lookahead curvature
        # calculate stanley steering
        # calculate velocity using lookahead curvature
        # apply steering and velocity desired control
        # close control loop over these??

    @staticmethod
    def calc_dead_reckoning(car_pos, car_speed, heading, yaw_rate, delta_time):
        updated_heading = heading + delta_time * yaw_rate
        updated_pos = car_pos + delta_time * car_speed * np.array([np.cos(updated_heading), np.sin(updated_heading)])
        return updated_pos, updated_heading


if __name__ == '__main__':
    # connect to the AirSim simulator
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    print("API Control enabled: %s" % client.isApiControlEnabled())
    car_controls = airsim.CarControls()
    car_state = client.getCarState()
    car_state.speed
    car_state.kinematics_estimated.position
    car_state.kinematics_estimated.orientation
