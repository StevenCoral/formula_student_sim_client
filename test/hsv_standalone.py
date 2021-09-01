import numpy as np
import pickle
from matplotlib import pyplot as plt
import camera_utils
import spatial_utils
import tracker_utils
import cv2
import time
import os


image_dest = 'D:\\Msc_Project\\BGR_client\\images'
left_image = os.path.join(image_dest, 'left_1.png')
left_cv_img = cv2.imread(left_image)
left_hsv = cv2.cvtColor(left_cv_img, cv2.COLOR_BGR2HSV)
right_image = os.path.join(image_dest, 'right_1.png')
right_cv_img = cv2.imread(right_image)
right_hsv = cv2.cvtColor(right_cv_img, cv2.COLOR_BGR2HSV)

cv2.imwrite(os.path.join(image_dest, 'left_h.png'), left_hsv[:, :, 0])
cv2.imwrite(os.path.join(image_dest, 'left_s.png'), left_hsv[:, :, 1])
cv2.imwrite(os.path.join(image_dest, 'right_h.png'), right_hsv[:, :, 0])
cv2.imwrite(os.path.join(image_dest, 'right_s.png'), right_hsv[:, :, 1])

blue = np.uint8([[[255, 0, 0]]])
hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
print('blue:', hsv_blue)

green = np.uint8([[[0, 255, 0]]])
hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
print('green:', hsv_green)

red = np.uint8([[[0, 0, 255]]])
hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
print('red:', hsv_red)

orange = np.uint8([[[0, 165, 255]]])
hsv_orange = cv2.cvtColor(orange, cv2.COLOR_BGR2HSV)
print('orange:', hsv_orange)

yellow = np.uint8([[[0, 255, 255]]])
hsv_yellow = cv2.cvtColor(yellow, cv2.COLOR_BGR2HSV)
print('yellow:', hsv_yellow)

left_cam = camera_utils.AirsimCamera(640, 360, 80, [2, -0.5, -0.5], [-40.0, -10.0, 0])
right_cam = camera_utils.AirsimCamera(640, 360, 80, [2, 0.5, -0.5], [40.0, -10.0, 0])

hsv_img = left_hsv
h_channel = hsv_img[:, :, 0]
s_channel = hsv_img[:, :, 1]

cv2.imshow('H', h_channel)
cv2.imshow('S', s_channel)
cv2.waitKey(1)
a=1
