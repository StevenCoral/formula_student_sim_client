import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
import tracker_utils


class PathSpline:
    def __init__(self, x_points, y_points):
        self.xb = x_points  # Base points' x values
        self.yb = y_points  # Base points' y values
        self.xi = None  # Interpolated points' x values
        self.yi = None  # Interpolated points' y values
        self.curvature = None  # Menger curvature of interpolated data
        self.meters_per_index = 0.0
        self.path_length = 0.0  # meters
        self.array_length = int(0)
        self.last_idx = int(0)

    def generate_spline(self, amount, meters=True, smoothing=0, summation=10000):
        # High-level API function to be used for generating a path spline.
        # If meters is True, amount will be roughly the distance between 2 points (indices),
        # Otherwise, amount is the actual number of indices the spline will have.

        # Measure the path length using "summation" number of points regardless of the "amount" choice:
        x_temp, y_temp, c_temp = self._calculate_spline(summation, smoothing)
        total_length = 0.0
        x_diff = np.diff(x_temp)
        y_diff = np.diff(y_temp)
        for idx in range(summation - 1):
            total_length += np.sqrt(x_diff[idx] ** 2 + y_diff[idx] ** 2)

        if meters:
            num_idx = int(total_length / amount)
            self.meters_per_index = total_length / float(num_idx)
        else:
            num_idx = int(amount)
            self.meters_per_index = total_length / amount

        self.xi, self.yi, self.curvature = self._calculate_spline(num_idx, smoothing)

        self.array_length = len(self.xi)
        self.last_idx = self.array_length - 1
        self.path_length = total_length

    def _calculate_spline(self, num_idx, smoothing=0):
        # Fits splines to x=f(u) and y=g(u), treating both as periodic. Also note that s=0
        # is used only to force the spline fit to pass through all the input points.
        tck, u = interpolate.splprep([self.xb, self.yb], s=smoothing, per=True, quiet=1)

        # Evaluate the spline fits for 10000 evenly spaced distance values:
        xi, yi = interpolate.splev(np.linspace(0, 1, num_idx), tck)

        # Evaluate the curvature for each and every point:
        ci = np.zeros(num_idx)
        for idx in range(num_idx - 2):
            if idx == 0:
                pass
            else:
                first_p = np.array([xi[idx - 1], yi[idx - 1]])
                second_p = np.array([xi[idx], yi[idx]])
                third_p = np.array([xi[idx + 1], yi[idx + 1]])
                ci[idx] = self.menger_curvature(first_p, second_p, third_p)

        ci[0] = ci[1]
        ci[-1] = ci[-2]

        return xi, yi, ci

    def find_closest_point(self, target_point):
        # Finding the closest point using "brute force", no optimization.
        # Minimum for norm will be minimum for norm squared as well, dropping the sqrt():
        distances = (self.xi - target_point[0])**2 + (self.yi - target_point[1])**2
        closest_idx = distances.argmin()
        if closest_idx == 0:
            prev_idx = self.last_idx
            next_idx = closest_idx + 1
        elif closest_idx == self.last_idx:
            prev_idx = closest_idx - 1
            next_idx = 0
        else:
            prev_idx = closest_idx - 1
            next_idx = closest_idx + 1

        closest_vector = np.array([self.xi[closest_idx] - target_point[0], self.yi[closest_idx] - target_point[1]])
        closest_tangent = np.array([self.xi[next_idx] - self.xi[prev_idx], self.yi[next_idx] - self.yi[prev_idx]])
        if np.linalg.norm(closest_tangent) > 1e-6:
            closest_tangent /= np.linalg.norm(closest_tangent)  # Normalize to a length of 1
        # else ?

        return closest_idx, closest_vector, closest_tangent

    @staticmethod
    def menger_curvature(prev_p, curr_p, next_p):
        prev_unit_vec = (curr_p - prev_p) / np.linalg.norm(curr_p - prev_p)
        next_unit_vec = (next_p - curr_p) / np.linalg.norm(next_p - curr_p)
        base_angle = np.arccos(np.dot(prev_unit_vec, next_unit_vec))
        base_length = np.linalg.norm(next_p - prev_p)
        curvature = 2 * np.sin(base_angle) / base_length
        if curvature < 1e-4:  # Meaningless value for our purpose
            curvature = 0
        return curvature


def generate_path_points(mapping_data):
    # Uses the list of tracked cones in order to generate a valid spline:
    left_points = np.ndarray(shape=(0, 2))
    right_points = np.ndarray(shape=(0, 2))
    unknown_points = np.ndarray(shape=(0, 2))
    for tracked_obj in mapping_data:
        if tracked_obj.active:
            point = tracked_obj.position[0:2]
            if tracked_obj.color == tracker_utils.ConeTracker.COLOR_BLUE:
                left_points = np.append(left_points, point.reshape(1, 2), axis=0)
            elif tracked_obj.color == tracker_utils.ConeTracker.COLOR_YELLOW:
                right_points = np.append(right_points, point.reshape(1, 2), axis=0)
            else:
                # Qualify unknown cones as the closest color (potentially dangerous):
                left_dist = np.linalg.norm(left_points - point, axis=1)
                right_dist = np.linalg.norm(right_points - point, axis=1)
                if np.min(left_dist) < np.min(right_dist):
                    left_points = np.append(left_points, point.reshape(1, 2), axis=0)
                else:
                    right_points = np.append(right_points, point.reshape(1, 2), axis=0)
                unknown_points = np.append(unknown_points, point.reshape(1, 2), axis=0)

    # Assuming somewhat equal cone detections on each side!
    min_length = min(left_points.shape[0], right_points.shape[0])
    track_points = (left_points[:min_length, :] + right_points[:min_length, :]) / 2

    return track_points

