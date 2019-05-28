#!/usr/bin/env python

import rospy
import numpy as np
import cv2
import time
import math
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


def test_match(map_name, image_name, tolerance, black_threshold):
    im_model_raw = cv2.imread(map_name, cv2.IMREAD_GRAYSCALE)
    _, im_model = cv2.threshold(
        im_model_raw, 1, 255, cv2.THRESH_BINARY)

    im_raw = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    _, im = cv2.threshold(im_raw, black_threshold, 255, cv2.THRESH_BINARY)

    points_model = read_image(im_model)
    points = read_image(im)

    np.random.shuffle(points_model)
    extension = points.shape[0] - points_model.shape[0]
    repetitions = math.ceil(float(extension)/len(points_model))
    points_model_upsampled = np.append(
        points_model, np.repeat(points_model, repetitions, axis=0)[0:extension, :], axis=0)

    T, distances, iterations = icp.icp(
        points, points_model_upsampled, tolerance=tolerance)

    # Make C a homogeneous representation of B
    points_transformed = np.ones((points.shape[0], 4))
    points_transformed[:, 0:3] = points

    # Transform C
    points_transformed = np.dot(T, points_transformed.T).T

    # finally calculate free area difference
    _, im_free = cv2.threshold(im_raw, 240, 255, cv2.THRESH_BINARY)
    cv2.imshow('image', im_free)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    white_pixels = np.sum(im_free == 255)
    white_pixels_model = np.sum(im_model_raw == 255)
    white_space_error = white_pixels_model - white_pixels
    print('Free space mapping error', float(white_space_error)/white_pixels_model*100, '%')

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
    parser.add_argument('-b', '--threshold', type=int,
                        help='Binarize black threshold (default is 1).', default=50)
    args, _ = parser.parse_known_args()

    test_match(args.map, args.image, args.tolerance, args.threshold)
