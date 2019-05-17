#!/usr/bin/env python

import rospy
import numpy as np
import cv2
import time
import icp
import argparse
import matplotlib.pyplot as plt


def read_image(im):
    height, width = im.shape
    black_pixels = np.sum(im == 0)

    points = np.zeros((black_pixels, 3))

    count = 0
    for i in range(0, height):
        for j in range(0, width):
            if im[i, j] == 0:
                points[count, :] = np.array([i, j, 0])
                count = count + 1

    return points


def test_match(map_name, image_name, tolerance):
    im_model_raw = cv2.imread(map_name, cv2.IMREAD_GRAYSCALE)
    _, im_model = cv2.threshold(
        im_model_raw, 254, 255, cv2.THRESH_BINARY)

    im_raw = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    _, im = cv2.threshold(im_raw, 1, 255, cv2.THRESH_BINARY)

    points_model = read_image(im_model)
    points = read_image(im)

    np.random.shuffle(points_model)
    extension = points.shape[0] - points_model.shape[0]
    points_model_upsampled = np.append(
        points_model, points_model[0:extension, :], axis=0)

    T, distances, iterations = icp.icp(
        points, points_model_upsampled, tolerance=tolerance)

    # Make C a homogeneous representation of B
    points_transformed = np.ones((points.shape[0], 4))
    points_transformed[:, 0:3] = points

    # Transform C
    points_transformed = np.dot(T, points_transformed.T).T

    print('Transformation Matrix:', T)

    print('Point distances:', distances)

    print('Pixel squared error:', np.average(distances ** 2))

    print('Number of iterations:', iterations)

    plt.plot(points_transformed[:, 0], points_transformed[:, 1], 'bo')
    plt.plot(points_model[:, 0], points_model[:, 1], 'ro')
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    rospy.init_node('icp_map_comparison_node')

    parser = argparse.ArgumentParser(
        description='Calculates the minimal transformation between two maps.')
    parser.add_argument('-m', '--map', type=str,
                        help='The input map name.')
    parser.add_argument('-i', '--image', type=str,
                        help='The input image name.')
    parser.add_argument('-t', '--tolerance', type=float,
                        help='ICP tolerance (default is 0.0001).', default=0.0001)
    args, _ = parser.parse_known_args()

    test_match(args.map, args.image, args.tolerance)