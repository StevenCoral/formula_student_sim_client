import cone_mapping
import path_following
import airsim
import spline_utils
import numpy as np

# Create an airsim client instance:
airsim_client = airsim.CarClient()
airsim_client.confirmConnection()
airsim_client.enableApiControl(True)

# Detect the cones and spline points, and return their location:
mapping_data = cone_mapping.mapping_loop(airsim_client)

# Stop until spline generation is complete:
car_controls = airsim_client.getCarControls()
car_controls.throttle = 0.0
airsim_client.setCarControls(car_controls)

# Arrange the points and generate a path spline:
pursuit_points = np.ndarray(shape=(0, 2))
for tracked_obj in mapping_data:
    pursuit_points = np.append(pursuit_points, tracked_obj[:2].reshape(1, 2), axis=0)
spline_obj = spline_utils.PathSpline(pursuit_points[::2, 0], pursuit_points[::2, 1])
spline_obj.generate_spline(amount=0.1, meters=True, smoothing=1)

# Follow the spline using Stanley's method:
path_following.following_loop(airsim_client, spline_obj)
