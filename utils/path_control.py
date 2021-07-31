from spline_utils import PathSpline
import numpy as np
import airsim
from scipy.spatial.transform import Rotation as Rot
import spatial_utils
import time
import pickle


class StanleyFollower:
    EPSILON = 1e-4  # For numerical stability

    def __init__(self, path_spline):

        self.path = path_spline

        # Max steering MUST be the same as your simulated vehicle's settings within Unreal.
        self.max_velocity = 10.0  # m/s
        self.min_velocity = 5.0  # m/s
        self.max_steering = np.deg2rad(40.0)  # radians

        # Generic value:
        self.k_vel = (self.max_velocity - self.min_velocity) / (self.path.curvature.max() + self.EPSILON)
        self.lookahead = 5.0  # meters
        self.k_steer = 10.0  # Stanley steering coefficient

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
        theta_f = -np.arctan(self.k_steer * closest_dist**2 / (np.abs(car_vel) + self.EPSILON))*np.sign(ego_closest[1])

        # Combine them together and saturate:
        steering_angle = np.clip(theta_e + theta_f, -self.max_steering, self.max_steering)

        # Then we handle the desired speed [m/s]:
        lookahead_idx = closest_idx + int(np.around(self.lookahead / (self.path.meters_per_index + self.EPSILON)))
        if lookahead_idx >= self.path.array_length:
            lookahead_idx = lookahead_idx - self.path.array_length

        speed = self.max_velocity - self.k_vel * self.path.curvature[lookahead_idx]
        speed = max(speed, self.min_velocity)

        return speed, steering_angle


class PursuitFollower:
    EPSILON = 1e-4  # For numerical stability

    def __init__(self, min_distance, max_distance):

        self.min_lookahead = min_distance
        self.max_lookahead = max_distance
        self.previous_angle = 0.0
        self.k_steer = 0.5
        self.max_steering = np.deg2rad(40.0)  # Radians
        self.pursuit_points = [np.array([0.0, 0.0, 0.0])]

    def calc_ref_steering(self, tracked_cones, map_to_vehicle):
        # Manage pursuit points:
        last_blue = None
        last_yellow = None
        for curr_cone in reversed(tracked_cones):
            if curr_cone.active:
                if last_blue is None and curr_cone.color == curr_cone.COLOR_BLUE:
                    last_blue = curr_cone.position
                if last_yellow is None and curr_cone.color == curr_cone.COLOR_YELLOW:
                    last_yellow = curr_cone.position
                if last_blue is not None and last_yellow is not None:
                    last_average = (last_blue + last_yellow) / 2.0
                    pursuit_diff = np.linalg.norm(self.pursuit_points[-1] - last_average)
                    # Only add the current point to the list if it is not the same as the last one:
                    if pursuit_diff > 1.0:
                        self.pursuit_points.append(last_average)
                    break

        # Scanning pursuit points from newest (and farthest) backwards.
        # If the point is farther than the max lookahead distance, keep scanning.
        # If we reached a point that is too close, keep moving straight until (hopefully) a new one gets within range.
        if len(self.pursuit_points) > 2:
            for point in reversed(self.pursuit_points):
                # Compute pursuit point in vehicle coordinates:
                pursuit_map = np.append(point, 1)
                pursuit_vehicle = np.matmul(map_to_vehicle, pursuit_map)[:3]
                pursuit_distance = np.linalg.norm(pursuit_vehicle)
                if pursuit_distance < self.max_lookahead:
                    break
            if pursuit_distance > self.min_lookahead:
                steering_angle = -self.k_steer * np.arctan(pursuit_vehicle[1] / (pursuit_vehicle[0] + self.EPSILON))
                self.previous_angle = np.clip(steering_angle, -self.max_steering, self.max_steering)
            else:
                steering_angle = 0.0
        else:
            steering_angle = 0.0  # Move straight if we haven't picked up any points yet.

        steering_angle = np.clip(steering_angle, -self.max_steering, self.max_steering)

        return steering_angle


class ProximitySteer:  #TODO remove
    def __init__(self, front=1.0, sideways=1.0):
        self.front_distance = front
        self.sideways_distance = sideways
        self.max_steering = np.deg2rad(40.0)  # Radians
        self.steering_coeff = 1.0

    def calculate_steering(self, detections):
        numpy_dets = np.asarray(detections)  # To support also lists
        left_detections = self.filter_side(numpy_dets, True)
        right_detections = self.filter_side(numpy_dets, False)
        left_min = np.min(np.linalg.norm(left_detections))
        right_min = np.min(np.linalg.norm(right_detections))
        steering = self.steering_coeff * (left_min - right_min)
        direction = 'left' if steering > 0 else 'right'
        print(direction, steering)
        return steering

    def filter_side(self, detections, left_side):
        if left_side:
            sideways_filter = np.bitwise_and(detections[:, 1] > 0.0, detections[:, 1] < self.sideways_distance)
        else:  # Only left or right...
            sideways_filter = np.bitwise_and(detections[:, 1] < 0.0, detections[:, 1] > -self.sideways_distance)

        front_filter = np.bitwise_and(detections[:, 0] > 0.0, detections[:, 0] < self.front_distance)
        filtered_indices = np.bitwise_and(sideways_filter, front_filter)

        return detections[filtered_indices, :]

    # How do we get the trackers back in vehicle frame?
    # Append raw lidar detections to a new list if they correspond to active trackers?


def calc_dead_reckoning(car_pos, car_speed, heading, yaw_rate, delta_time):
    updated_heading = heading + delta_time * yaw_rate
    updated_pos = car_pos + delta_time * car_speed * np.array([np.cos(updated_heading), np.sin(updated_heading)])
    return updated_pos, updated_heading

if __name__ == '__main__':

    # steer_controller = PidfControl(0.01)
    # steer_controller.set_pidf(900.0, 0.0, 42.0, 0.0)
    # steer_controller.set_extrema(0.01, 1.0)
    # steer_controller.alpha = 0.01
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

    x = np.array([10.00, 10.00, 10.00, 10.00, 10.00, 6.00, -6.00, -19.00, -23.00, -23.00, -17.00, 0.0, 8.00])
    y = np.array([20.00, 10.00, -10.00, -40.00, -60.00, -73.00, -78.00, -70.00, -38.00, 10.00, 30.00, 31.00, 27.00])

    x -= starting_x
    y -= starting_y

    my_spline = PathSpline(x, y)
    my_spline.generate_spline(0.1, smoothing=1)
    follow_handler = StanleyFollower(my_spline)
    follow_handler.k_vel *= 2.0

