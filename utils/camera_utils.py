import numpy as np
import spatial_utils
import cv2
from matplotlib import pyplot as plt


class AirsimCamera:
    HUE_BLUE = 109
    SAT_MIN_YELLOW = 120
    SAT_MAX_YELLOW = 160
    HUE_YELLOW = 27
    SAT_MIN_BLUE = 200
    SAT_MAX_BLUE = 240

    # cam_pos and cam_rot should be given in AIRSIM coordinates, as specified in settings.json!
    def __init__(self, cam_width, cam_height, cam_fov, cam_pos=np.array([0, 0, 0]), cam_rot=np.array([0, 0, 0])):
        self.width = cam_width
        self.height = cam_height
        self.fov = cam_fov
        self.pos = cam_pos
        self.rot = cam_rot
        self.intrinsic_matrix = self.generate_intrinsics(cam_width, cam_height, cam_fov)
        # Transformation matrix is calculated in ENG coordinate system!
        self.tf_matrix = spatial_utils.tf_matrix_from_airsim_pose(cam_pos, cam_rot)

    @staticmethod
    def generate_intrinsics(width, height, horizontal_fov):
        focal_distance = 0.5 * width / (np.tan(np.deg2rad(horizontal_fov) / 2))
        cam_intrinsics = np.zeros(shape=(3, 3))
        cam_intrinsics[0, 0] = focal_distance
        cam_intrinsics[1, 1] = focal_distance
        cam_intrinsics[2, 2] = 1.0
        cam_intrinsics[0, 2] = width / 2
        cam_intrinsics[1, 2] = height / 2
        return cam_intrinsics

    def project_vector_to_pixel(self, vector):
        cam_coordinates = spatial_utils.eng_to_camera(np.array(vector))
        distance = np.linalg.norm(cam_coordinates)
        normalized_vector = cam_coordinates / cam_coordinates[2]
        pixel_space = np.matmul(self.intrinsic_matrix, normalized_vector)
        # Note that camera pixel space axes are not the same as matrix index representation!
        return pixel_space[:2], distance

    def generate_cropping_indices(self, vector):
        pixel, dist = self.project_vector_to_pixel(vector)
        pixel = np.round(pixel).astype(np.int32)
        numpy_indices = np.flip(pixel)
        rect_h = np.round(40 / dist).astype(np.int32)  # Heuristic, half-size of desired rectangle height
        rect_w = np.round(30 / dist).astype(np.int32)  # Heuristic, half-size of desired rectangle width
        h_range = [numpy_indices[0] - rect_h, numpy_indices[0] + rect_h]
        w_range = [numpy_indices[1] - rect_w, numpy_indices[1] + rect_w]
        return h_range, w_range

    def get_cropped_hsv(self, image, vector):
        h_range, w_range = self.generate_cropping_indices(vector)
        h_range = np.clip(h_range, 0, image.shape[0])
        w_range = np.clip(w_range, 0, image.shape[1])
        if h_range[1] <= h_range[0] or w_range[1] <= w_range[0]:
            return np.empty(shape=(1, 1)), False
        else:
            return cv2.cvtColor(image[h_range[0]:h_range[1], w_range[0]:w_range[1], :], cv2.COLOR_BGR2HSV), True


def get_bgr_image(airsim_response):
    # Acquire and reshape image:
    image = np.frombuffer(airsim_response.image_data_uint8, dtype=np.uint8).reshape(airsim_response.height,
                                                                                    airsim_response.width,
                                                                                    3)
    return image


def save_img(airsim_response, file_path):
    # Acquire and reshape image:
    image = get_bgr_image(airsim_response)

    # Write to file
    cv2.imwrite(file_path, image)


def dump(image):
    pass

    # pixel, dist = left_cam.project_vector_to_pixel(curr_centroid)
    # pixel = np.round(pixel).astype(np.uint32)
    # indices = np.flip(pixel)
    # rect_h = np.round(40 / dist).astype(np.uint32)  # Heuristic, half-size of desired rectangle height
    # rect_w = np.round(30 / dist).astype(np.uint32)  # Heuristic, half-size of desired rectangle width
    # cropped_img = left_cv_img[indices[0]-rect_h:indices[0]+rect_h, indices[1]-rect_w:indices[1]+rect_w, :]
    # # cropped_img = right_cv_img[250:260, 170:176, :]
    # hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    # hsv_size = hsv_img.shape[0] * hsv_img.shape[1]
    # hue_histogram = cv2.calcHist(hsv_img, [0], None, [180], [0, hsv_size])
    # plt.hist(hsv_img[:, :, 0].flatten(), 180)
    # plt.show()
    # rect_extent = np.array([rect_w, rect_h], dtype=np.uint32)
    # left_cv_img = cv2.rectangle(left_cv_img, pixel-rect_extent, pixel+rect_extent, (0, 0, 255), 1)