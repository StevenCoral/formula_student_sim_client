from scipy.spatial.transform import Rotation
import numpy as np


bottom_row = np.array([0.0, 0.0, 0.0, 1.0]).reshape(1, 4)


def set_initial_pose(airsim_client, desired_position, desired_heading):
    initial_pose = airsim_client.simGetVehiclePose()
    rot = Rotation.from_euler('xyz', [0, 0, desired_heading], degrees=True)
    quat = rot.as_quat()
    initial_pose.orientation.x_val = quat[0]
    initial_pose.orientation.y_val = quat[1]
    initial_pose.orientation.z_val = quat[2]
    initial_pose.orientation.w_val = quat[3]
    initial_pose.position.x_val = desired_position[0]
    initial_pose.position.y_val = desired_position[1]
    # initial_pose.position.z_val = desired_position[2]
    airsim_client.simSetVehiclePose(initial_pose, ignore_collison=True)


def extract_position(pos, pos_offset=None):
    if pos_offset is None:
        offset_x = 0.0
        offset_y = 0.0
    else:
        offset_x = pos_offset[0]
        offset_y = pos_offset[1]
    return np.array([pos.x_val + offset_x, pos.y_val + offset_y, pos.z_val])


def extract_rotation(orientation):
    quaternion = np.array([orientation.x_val,
                           orientation.y_val,
                           orientation.z_val,
                           orientation.w_val])
    return Rotation.from_quat(quaternion)


def extract_pose(actor_pose, pos_offset=None):
    position = extract_position(actor_pose.position, pos_offset)
    speed = actor_pose.speed
    rotation = extract_rotation(actor_pose.orientation)
    heading = rotation.as_euler('xyz', degrees=False)[2]
    return position, speed, heading, rotation


def extract_transform_matrix(actor_pose, pos_offset=None):
    position = extract_position(actor_pose.position, pos_offset)
    rotation = extract_rotation(actor_pose.orientation).as_matrix()
    tf_matrix = np.append(rotation, position.reshape(3, 1), axis=1)
    tf_matrix = np.append(tf_matrix, bottom_row, axis=0)

    return tf_matrix


if __name__ == "__main__":
    import airsim
    test_pose = airsim.Pose()
    tf_mat = extract_transform_matrix(test_pose, [12, 34])
    pass