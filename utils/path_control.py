from spline_utils import PathSpline
import numpy as np
import airsim
from scipy.spatial.transform import Rotation as Rot
import spatial_utils
import time
import pickle


class PathFollower:
    def __init__(self, path_spline):

        self.path = path_spline

        # The rest of the parameters should be set post-initialization,
        # since it is usually less frequently changed (vehicle params):
        self.max_velocity = 10.0  # m/s
        self.min_velocity = 6.0  # m/s
        self.max_steering = np.deg2rad(40.0)  # radians
        self.epsilon = 1e-4  # For numerical stability

        # Generic value:
        self.k_vel = (self.max_velocity - self.min_velocity) / (self.path.curvature.max() + self.epsilon)
        self.lookahead = 5.0  # meters
        self.k_steer = 10.0  # Stanley steering coefficient
        self.stanley_denom = 1e-4

    def calc_ref_speed_steering(self, car_pos, car_vel, heading):
        # First we match the path steering angle [rad]:
        closest_idx, closest_vector, closest_tangent = self.path.find_closest_point(car_pos)
        path_direction = np.arctan2(closest_tangent[1], closest_tangent[0])
        theta_e = heading - path_direction
        if theta_e > 1.5 * np.pi:
            theta_e -= 2.0 * np.pi
        if theta_e < -1.5 * np.pi:
            theta_e += 2.0 * np.pi

        # Second we calculate the closing steering angle [rad]:
        closest_dist = np.linalg.norm(closest_vector)
        ego_rotation = np.array([[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]])
        ego_closest = np.matmul(ego_rotation, closest_vector)  # Closest vector in vehicle frame of reference
        theta_f = -np.arctan(self.k_steer * closest_dist**2 / (np.abs(car_vel) + self.epsilon)) * \
                  np.sign(ego_closest[1])
        steering_angle = np.clip(theta_e + theta_f, -self.max_steering, self.max_steering)

        # Then we handle the desired speed [m/s]:
        lookahead_idx = closest_idx + int(np.around(self.lookahead / (self.path.meters_per_index + self.epsilon)))
        if lookahead_idx >= self.path.array_length:
            lookahead_idx = lookahead_idx - self.path.array_length

        speed = self.max_velocity - self.k_vel * self.path.curvature[lookahead_idx]
        speed = max(speed, self.min_velocity)

        return speed, steering_angle, closest_idx, closest_dist, closest_tangent, theta_e, theta_f

    @staticmethod
    def calc_dead_reckoning(car_pos, car_speed, heading, yaw_rate, delta_time):
        updated_heading = heading + delta_time * yaw_rate
        updated_pos = car_pos + delta_time * car_speed * np.array([np.cos(updated_heading), np.sin(updated_heading)])
        return updated_pos, updated_heading


if __name__ == '__main__':
    # Airsim is stupid, always spawns at zero. Must compensate using "playerstart" in unreal:
    starting_x = 10.0
    starting_y = 20.0

    x = np.array([10.00, 10.00, 10.00, 10.00, 10.00, 6.00, -6.00, -18.00, -23.00, -23.00, -17.00, 0.0, 8.00])
    y = np.array([20.00, 10.00, -10.00, -40.00, -60.00, -73.00, -78.00, -70.00, -38.00, 10.00, 30.00, 31.00, 27.00])

    x -= starting_x
    y -= starting_y

    my_spline = PathSpline(x, y)
    my_spline.generate_spline(0.1)
    follow_handler = PathFollower(my_spline)
    follow_handler.k_vel *= 2.0

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
