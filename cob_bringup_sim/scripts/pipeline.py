#!/usr/bin/env python

import rospy
import os
import psutil
import tf
import numpy as np
import geometry_msgs.msg
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Transform, TransformStamped
import matplotlib.pyplot as plt


class PoseComparison:

    def __init__(self):
        # list of poses and counter
        self.pose_history = list()
        self.i = 0
        # setup cpu logger to log slam effort
        self.cpu_history = dict()
        self.cpu_history['cpu'] = list()
        self.cpu_history['memory'] = list()
        # register shutdown hooks
        rospy.on_shutdown(self.shutdown)
        # tf utils
        self.tl = tf.TransformListener(rospy.Duration(5))
        self.br = tf.TransformBroadcaster()
        # subscribers
        rospy.Subscriber('/base_pose_ground_truth', Odometry, self.ground_truth_callback, queue_size=1)
        # wait for first transform to be published to set up tf callback
        self.tl.waitForTransform('map', 'ground_truth', rospy.Time(), rospy.Duration(5.0))
        rospy.Subscriber('/tf', TFMessage, self.tf_callback, queue_size=1)
        # store last pose
        self.last_pose = Transform()

        # all nodes are named /slam_mapping so look for pid
        self.slam_pid = int(os.popen('rosnode info /slam_mapping 2>/dev/null | grep Pid| cut -d\' \' -f2').read())
        self.slam_process = psutil.Process(pid=self.slam_pid)
        rospy.Timer(rospy.Duration(0.1), self.cpu_logger)

    def cpu_logger(self, event):
        self.cpu_history['cpu'].append(self.slam_process.cpu_percent())
        self.cpu_history['memory'].append(self.slam_process.memory_full_info().uss)

    def ground_truth_callback(self, odometry):
        # Function that republishes ground truth data as a tf with extrapolation
        self.ground_truth = TransformStamped()
        self.ground_truth.header = odometry.header
        self.ground_truth.child_frame_id = 'ground_truth'
        self.ground_truth.transform.translation.x = odometry.pose.pose.position.x
        self.ground_truth.transform.translation.y = odometry.pose.pose.position.y
        self.ground_truth.transform.translation.z = odometry.pose.pose.position.z
        self.ground_truth.transform.rotation = odometry.pose.pose.orientation
        self.br.sendTransformMessage(self.ground_truth)

    def tf_callback(self, tf_message):
        for transform in tf_message.transforms:
            if transform.header.frame_id == 'map' and transform.child_frame_id == 'odom_combined':
                # check if pose was updated by slam algorithm
                if not transform.transform == self.last_pose:
                    # calculate extrapolation of ground truth in that point
                    self.tl.waitForTransform('map', 'ground_truth', transform.header.stamp, rospy.Duration(1))
                    self.tl.waitForTransform('map', 'base_footprint', transform.header.stamp, rospy.Duration(1))
                    # push pose into pose history (ground truth and slam)
                    pose = dict()
                    pose['ground_truth'] = self.tl.lookupTransform('map', 'ground_truth', transform.header.stamp)
                    pose['slam'] = self.tl.lookupTransform('map', 'base_footprint', transform.header.stamp)
                    self.pose_history.append(pose)
                    rospy.loginfo('Pose #{}\n{}'.format(self.i, pose))
                    # finally update pose and increment pose count
                    self.last_pose = transform.transform
                    self.i = self.i + 1

    def shutdown(self):
        vertices = np.zeros((len(self.pose_history), 4))
        angles = np.zeros((len(self.pose_history), 2))

        for index, entry in enumerate(self.pose_history):
            vertices[index, :2] = entry['ground_truth'][0][0:2]
            vertices[index, 2:] = entry['slam'][0][0:2]
            angles[index, 0] = tf.transformations.euler_from_quaternion(entry['ground_truth'][1])[2]
            angles[index, 1] = tf.transformations.euler_from_quaternion(entry['slam'][1])[2]
        print('Vertices are', vertices)
        print('Angles are', angles)
        plt.axis('equal')
        plt.scatter(vertices[:, 0], vertices[:, 1])
        plt.scatter(vertices[:, 2], vertices[:, 3], c='red')
        plt.show()
        print('CPU Usage', self.cpu_history)


if __name__ == "__main__":
    rospy.init_node('pipeline_node')

    try:
        a = PoseComparison()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass