import numpy as np
from scipy.spatial.transform import Rotation as Rot
import time
import pickle
import csv
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN


def filter_cloud(pointcloud, min_distance=0.0, max_distance=100.0, min_height=-10.0, max_height=10.0):
    distances = np.linalg.norm(pointcloud, axis=1)
    filtered_distance = np.bitwise_and(distances > min_distance,
                                       distances < max_distance)
    filtered_heights = np.bitwise_and(pointcloud[:, 2] > min_height,
                                      pointcloud[:, 2] < max_height)
    filtered_indices = np.bitwise_and(filtered_distance, filtered_heights)

    return pointcloud[filtered_indices, :]


def cluster_extent(cluster):
    # Returns an array of the bounding-box extents of the cluster along the x and y axes.
    return np.amax(cluster, axis=0) - np.amin(cluster, axis=0)


def collate_segmentation(dbscan_obj, max_extent=0.5):
    # core_samples_mask = np.zeros_like(dbscan_obj.labels_, dtype=bool)
    # core_samples_mask[dbscan_obj.core_sample_indices_] = True

    # Sort-out labels:
    labels = dbscan_obj.labels_
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)  # No use for noise (label = -1)

    # Separate data into segments:
    segments = {}
    centroids = {}
    for idx in unique_labels:
        # class_member_mask = (labels == idx)
        # group = filtered_pc[class_member_mask & core_samples_mask]
        class_member_mask = (labels[dbscan_obj.core_sample_indices_] == idx)
        group = dbscan_obj.components_[class_member_mask]
        extents = cluster_extent(group)
        if group.size > 0 and np.linalg.norm(extents) < max_extent:
            segments[idx] = group
            centroids[idx] = np.mean(group, axis=0)

    return segments, centroids


