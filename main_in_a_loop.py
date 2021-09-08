import cone_mapping
import path_following
import airsim
import spline_utils
import path_control
import os
import pickle
import csv
import time

if __name__ == '__main__':

    # Create an airsim client instance:
    steering_procedure_manager = path_control.SteeringProcManager()
    airsim_client = airsim.CarClient()
    airsim_client.confirmConnection()
    airsim_client.enableApiControl(True)

    for run_idx in range(10):
        data_dest = os.path.join(os.getcwd(), 'recordings', 'recording' + str(run_idx))
        os.makedirs(data_dest)

        # Detect the cones and spline points, and return their location:
        print('Starting on-the-fly cone mapping with constant speed and steering procedure.')
        mapping_data, pursuit_points = cone_mapping.mapping_loop(airsim_client)
        print('Mapping complete!')

        # Stop until spline generation is complete:
        print('Stopping vehicle and generating a path to follow...')
        car_controls = airsim_client.getCarControls()
        car_controls.throttle = 0.0
        airsim_client.setCarControls(car_controls)

        # Save mappind data
        tracked_objects = {'cones': mapping_data, 'pursuit': pursuit_points}
        with open(os.path.join(data_dest, 'mapping_session.pickle'), 'wb') as pickle_file:
            pickle.dump(tracked_objects, pickle_file)
        print('pickle saved')

        # Arrange the points and generate a path spline:
        track_points = spline_utils.generate_path_points(mapping_data)
        spline_obj = spline_utils.PathSpline(track_points[::2, 0], track_points[::2, 1])
        spline_obj.generate_spline(amount=0.1, meters=True, smoothing=1)
        print('Done!')

        # Follow the spline using Stanley's method:
        print('Starting variable speed spline following procedure.')
        pickling_objects, car_data = path_following.following_loop(airsim_client, spline_obj)
        print('Full process complete! stopping vehicle.')

        # Done! stop vehicle:
        car_controls = airsim_client.getCarControls()
        car_controls.throttle = 0.0
        car_controls.brake = 1.0
        airsim_client.setCarControls(car_controls)
        steering_procedure_manager.terminate_steering_procedure()

        # Save following data
        with open(os.path.join(data_dest, 'following_session.pickle'), 'wb') as pickle_file:
            pickle.dump(pickling_objects, pickle_file)
        print('saved pickle data')
        with open(os.path.join(data_dest, 'car_data.csv'), 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['x', 'y', 'heading', 'v_desired', 'v_delivered',
                             's_desired', 's_delivered', 'throttle'])
            writer.writerows(car_data)
        print('saved csv data')

