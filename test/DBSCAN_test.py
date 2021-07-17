#!/usr/bin/env python
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import time
import pickle
import csv
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN


def cluster_extent(cluster):
    # Returns an array of the bounding-box extents of the cluster along the x and y axes.
    return np.amax(cluster, axis=0) - np.amin(cluster, axis=0)


if __name__ == '__main__':

    with open('pointcloud.pickle', 'rb') as pickle_file:
        pointcloud = pickle.load(pickle_file)
    print('loaded pointcloud data')
    tic = time.time()
    pointcloud[:, 2] *= -1  # Z is reversed in airsim because of flying convention
    distances = np.linalg.norm(pointcloud, axis=1)
    filtered_indices = np.bitwise_and(distances < 10.0,
                                      distances > 5.0)
    filtered_indices = np.bitwise_and(filtered_indices, pointcloud[:, 2] > -0.5)
    filtered_pc = pointcloud[filtered_indices, 0:2]

    print('filtering takes:', time.time()-tic)
    tic = time.time()

    # Compute DBSCAN
    db = DBSCAN(eps=0.1, min_samples=3).fit(filtered_pc)
    print('DBSCAN takes:', time.time() - tic)
    tic = time.time()

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # db.components correspond to the original pc array
    # db.labels correspond to each point in the array, assigining it a "group index"
    # db.core_sample_indices are indices in pc array which have SOME affiliation with a cluster

    # Number of clusters in labels, ignoring noise if present.
    unique_labels = set(labels)
    n_clusters_ = len(unique_labels) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)

    groups = {}
    centroids = {}
    max_extent = 0.5  # Diameter of a cone. Any cluster larger than that will be disqualified.
    for idx in unique_labels:
        if idx != -1:
            class_member_mask = (labels == idx)
            group = filtered_pc[class_member_mask & core_samples_mask]  # From all pointcloud
            group2 = db.components_[labels[db.core_sample_indices_] == idx]  # From non-noise samples (components) only

            extents = cluster_extent(group)
            if np.linalg.norm(extents) < max_extent:
                groups[idx] = group
                centroids[idx] = np.mean(group, axis=0)

    print('segmentation takes:', time.time() - tic)
    tic = time.time()

    # Black removed and is used for noise instead.
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = filtered_pc[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = filtered_pc[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    a=5

    # Plot the result
    fig, ax = plt.subplots(1, 1)
    ax.plot(filtered_pc[:, 0], filtered_pc[:, 1], '*b')
    ax.grid(True)
    plt.xlim([0, 10])
    plt.ylim([-5, 5])
    fig.show()
    a = 5