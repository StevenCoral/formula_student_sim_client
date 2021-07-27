from scipy.spatial.transform import Rotation
import numpy as np

# Unreal has a left-handed coordinate system, with odd rotations.
# The order of rotation is an intrinsic yaw -> pitch -> roll schema.
# To transfer from Unreal to engineering notations, one must:
# flip the Y, yaw and pitch directions (and vice versa).
# In addition, Airsim flips the Z direction because it uses an aerial NED coordinate system,
# so this must also be taken into account.
# ENG or eng refers to an Engineering coordinate system (right-handed),
# with X pointing forward, Y pointing left and Z pointing up.
# Cascaded multiplications add up from the left-hand side.

bottom_row = np.array([0.0, 0.0, 0.0, 1.0]).reshape(1, 4)


# Convert between 2 coordinate systems and vice-vera are the same function!
# Input should be numpy arrays. Behavior with lists is undefined.
def convert_eng_unreal(pos, rot):
    pos[1] *= -1
    rot[:2] *= -1
    return pos, rot


def convert_eng_airsim(pos, rot):
    pos[1] *= -1
    pos[2] *= -1
    rot[:2] *= -1
    return pos, rot


def convert_unreal_airsim(pos, rot):
    pos[2] *= -1
    return pos, rot


def eng_to_camera(pos):
    # Converts from engineering to camera coordinate systems without messing with rotation matrix calculations.
    new_pos = np.array([-pos[1], -pos[2], pos[0]])
    return new_pos


def camera_to_eng(pos):
    new_pos = np.array([pos[2], -pos[0], -pos[1]])
    return new_pos


def set_airsim_pose(airsim_client, desired_position, desired_rot, inherit_z=True):
    # Input is in ENG coordinate system!
    # Converts to Airsim and sends to client.

    desired_position = np.array(desired_position)  # To accept lists as well.
    desired_rot = np.array(desired_rot)  # To accept lists as well.
    initial_pose = airsim_client.simGetVehiclePose()

    if inherit_z:
        desired_position = np.append(desired_position, -initial_pose.position.z_val)

    pos, rot = convert_eng_airsim(desired_position, desired_rot)
    rotator = Rotation.from_euler('ZYX', rot, degrees=True)
    quat = rotator.as_quat()
    initial_pose.orientation.x_val = quat[0]
    initial_pose.orientation.y_val = quat[1]
    initial_pose.orientation.z_val = quat[2]
    initial_pose.orientation.w_val = quat[3]
    initial_pose.position.x_val = pos[0]
    initial_pose.position.y_val = pos[1]
    initial_pose.position.z_val = pos[2]
    airsim_client.simSetVehiclePose(initial_pose, ignore_collison=True)


def extract_rotation_from_airsim(orientation):
    # Input should be a Quaternionr() object directly from Airsim.
    # Output is in Airsim coordinate system.
    quaternion = np.array([orientation.x_val,
                           orientation.y_val,
                           orientation.z_val,
                           orientation.w_val])
    return Rotation.from_quat(quaternion).as_euler('ZYX', degrees=True)


def extract_pose_from_airsim(actor_pose):
    # Input is an Airsim Pose() object.
    # Output is in ENG coordinate system.
    position = np.array([actor_pose.position.x_val, actor_pose.position.y_val, actor_pose.position.z_val])
    rotation = extract_rotation_from_airsim(actor_pose.orientation)
    pos, rot = convert_eng_airsim(position, rotation)
    return pos, rot


def tf_matrix_from_eng_pose(position, rotation):
    # Input is position and rotation objects in ENG coordinate system.
    # Output is a 4x4 transform matrix in ENG coordinate system.
    position = np.array(position)
    rot = Rotation.from_euler('ZYX', rotation, degrees=True).as_matrix()
    tf_matrix = np.append(rot, position.reshape(3, 1), axis=1)
    tf_matrix = np.append(tf_matrix, bottom_row, axis=0)
    return tf_matrix


def tf_matrix_from_unreal_pose(position, rotation):
    # Input is position and rotation objects in Unreal coordinate system.
    # Output is a 4x4 transform matrix in ENG coordinate system.
    yaw_pitch_roll = np.array(rotation)
    pos, rot = convert_eng_unreal(position, yaw_pitch_roll)
    return tf_matrix_from_eng_pose(pos, rot)


def tf_matrix_from_airsim_pose(position, rotation):
    # Input is position and rotation objects in Airsim coordinate system.
    # Output is a 4x4 transform matrix in ENG coordinate system.
    yaw_pitch_roll = np.array(rotation)
    pos, rot = convert_eng_airsim(position, yaw_pitch_roll)
    return tf_matrix_from_eng_pose(pos, rot)


def tf_matrix_from_airsim_object(actor_pose):
    # Input is an Airsim Pose() object.
    # Output is a 4x4 transform matrix in ENG coordinate system.
    position = np.array([actor_pose.position.x_val, actor_pose.position.y_val, actor_pose.position.z_val])
    yaw_pitch_roll = extract_rotation_from_airsim(actor_pose.orientation)
    pos, rot = convert_eng_airsim(position, yaw_pitch_roll)
    return tf_matrix_from_eng_pose(pos, rot)


if __name__ == "__main__":
    import airsim
    import time
    client = airsim.CarClient()
    client.confirmConnection()
    time.sleep(1.0)

    set_airsim_pose(client, [0.0, 20.0], [45.0, 0, 0])
    test_pose = client.simGetVehiclePose()
    tf_mat = tf_matrix_from_airsim_pose(test_pose)
    tf_2 = tf_matrix_from_eng_pose([1, 2, 3], 40, 20)
    pass
