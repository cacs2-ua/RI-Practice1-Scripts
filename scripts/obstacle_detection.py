#!/usr/bin/env python

# This node (implemented in Python) detects obstacles in the scene using the point cloud from a LiDAR sensor.
# The obstacles can be filtered using the height in Z axis given the XYZ coordinates of each point of the cloud.

# A template with "TODO" statements is provided to ease the node implementation.

from __future__ import print_function

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import numpy as np
from math import atan2, cos, sin, sqrt, pi

# Publisher definition
pub_obstacles = None
pub_freezone = None

# Define the height to consider the obstacles and the radius to detect them. Both variables
# are expressed in meters.

# -------------------------------------------------------------------------
# BLOCK 1: Obstacle detection parameters.
# The Velodyne sensor is above the floor, therefore the floor usually appears
# with negative z values in the Velodyne frame. A threshold around -0.45 m
# removes most floor points while keeping vertical obstacle points.
# -------------------------------------------------------------------------

#TODO
# Set the appropriate height and radius values to detect the obstacles surrounding the object.
# Note: Consider the LiDAR sensor is mounted on the robot at a certain height. The points below the sensor
# have a negative value

altura = -0.45
radio = 2.20

minimum_detection_radius = 0.45
free_zone_angular_resolution = 2.0 * pi / 180.0


def is_valid_point(x, y, z):
    # -------------------------------------------------------------------------
    # BLOCK 2: Numeric validity check.
    # This prevents NaN or infinite LiDAR values from entering the planner.
    # -------------------------------------------------------------------------

    return np.isfinite(x) and np.isfinite(y) and np.isfinite(z)


def filter_obstacles_function(point_cloud_in, altura):

    # This function receives a PointCloud2 ROS message as input and outputs a point cloud
    # containing the detected obstacles based on a given height.

    # Convert the PointCloud2 message to an array containing the x,y,z values.
    pc_data = pc2.read_points(point_cloud_in, field_names=("x", "y", "z"), skip_nans=True)

    # Array containing the parameters of the detected obstacles but initialized as void
    obstacles_points = []

    # For loop to add the points corresponding to detected obstacles
    # An obstacle is considered based on the object height.

    for point in pc_data:

        x, y, z = point

        if not is_valid_point(x, y, z):
            continue

        horizontal_distance = sqrt(x * x + y * y)

        #TODO
        # Add the points higher than "altura" to the "obstacles_points" array given x,y,z.
        # Note: The points added to "obstacle_points" are projected to the "altura" value, that is to say,
        # the obstacle coordinates (x,y,z) will change to (x,y,altura)

        # -------------------------------------------------------------------------
        # BLOCK 3: Height and radius filtering.
        # Points are considered obstacles if:
        #   1) they are higher than the floor-removal threshold;
        #   2) they are not too close to the robot body;
        #   3) they are inside the local detection radius.
        # -------------------------------------------------------------------------

        if z > altura and minimum_detection_radius <= horizontal_distance <= radio:

            # Add the point to the detected obstacles:
            obstacles_points.append([x, y, altura])

    return obstacles_points


def create_uniform_free_zone_ring(radio, altura):
    # -------------------------------------------------------------------------
    # BLOCK 4: Fallback free-zone ring.
    # If the LiDAR cloud does not provide enough points to build the ring, this
    # fallback creates a uniform circular ring around the robot.
    # -------------------------------------------------------------------------

    free_zone = []
    number_of_points = int((2.0 * pi) / free_zone_angular_resolution)

    for index in range(number_of_points):
        angle = -pi + index * free_zone_angular_resolution
        new_x = radio * cos(angle)
        new_y = radio * sin(angle)
        free_zone.append([new_x, new_y, altura])

    return free_zone


def free_zone_function(point_cloud_in, radio, altura):

    # This function receives as input a PointCloud2 ROS message and outputs a point cloud with
    # a given radius and height. This point cloud represents the area free of obstacles

    # Convert the PointCloud2 message to an array with x,y,z values
    pc_data = pc2.read_points(point_cloud_in, field_names=("x", "y", "z"), skip_nans=True)

    # Array containing the parameters of the point cloud representing the area free of obstacles
    free_zone = []

    used_angle_bins = set()

    # For loop to add the points corresponding to the areas free of obstacles
    # If no object is detected within a radius, the point cloud free of obstacles is generated

    for point in pc_data:
        x, y, z = point

        if not is_valid_point(x, y, z):
            continue

        distance_from_robot = sqrt(x * x + y * y)

        #TODO
        # Check if the point is within the radius given the x,y distance

        # -------------------------------------------------------------------------
        # BLOCK 5: Ring generation from valid LiDAR directions.
        # Each valid LiDAR direction is projected to a fixed-radius ring. Angular
        # binning avoids publishing thousands of duplicate points.
        # -------------------------------------------------------------------------

        if minimum_detection_radius <= distance_from_robot <= radio:

            # To create the radius of free obstacles, we need to know the angle of each point given its x,y coordinate.

            # Calculate the angle given its x,y coordinates (arcotangente)
            ang = atan2(y, x)

            angle_bin = int((ang + pi) / free_zone_angular_resolution)

            if angle_bin in used_angle_bins:
                continue

            used_angle_bins.add(angle_bin)

            # Calculate the new x,y coordinates given the angle and the radius
            new_x = radio * cos(ang)
            new_y = radio * sin(ang)

            # Add point to the ring with z value equal to the height
            free_zone.append([new_x, new_y, altura])

    if len(free_zone) < 30:
        free_zone = create_uniform_free_zone_ring(radio, altura)

    return free_zone


def publish_topics(obstacles_points, free_zone_points):

    # This function publishes the topics related to the obstacles and free areas

    # Definition of global variables
    global pub_obstacles, pub_freezone

    # Header definition, both topics should take the frame of the Velodyne sensor
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "blue/velodyne"

    # Definition of the messages to publish the obstacles and free areas
    obstacles_msg = pc2.create_cloud_xyz32(header, obstacles_points)
    free_zone_msg = pc2.create_cloud_xyz32(header, free_zone_points)

    # Publish messages
    pub_obstacles.publish(obstacles_msg)
    pub_freezone.publish(free_zone_msg)


def point_cloud_callback(msg):

    # This callback function receives as input the message containing the point cloud of type PointCloud2.
    # This message is sent to the functions that detect obstacles and areas of free obstacles.

    # Definition of global variables
    global altura, radio

    # Point cloud of detected objects created by the "filter_obstacles_function" function.
    obstacles_points = filter_obstacles_function(msg, altura)

    # Point cloud of free obstacles created by the "free_zone_function" function.
    free_zone_points = free_zone_function(msg, radio, altura)

    # Publish obstacles and free areas
    publish_topics(obstacles_points, free_zone_points)


def main():
    rospy.init_node('point_cloud_filter_node', anonymous=True)

    global pub_obstacles, pub_freezone
    global altura, radio, minimum_detection_radius, free_zone_angular_resolution

    # -------------------------------------------------------------------------
    # BLOCK 6: Runtime ROS parameters.
    # These parameters allow tuning without modifying the Python file.
    # -------------------------------------------------------------------------

    altura = rospy.get_param("~height_threshold", altura)
    radio = rospy.get_param("~detection_radius", radio)
    minimum_detection_radius = rospy.get_param("~minimum_detection_radius", minimum_detection_radius)

    angular_resolution_degrees = rospy.get_param("~free_zone_angular_resolution_degrees", 2.0)
    free_zone_angular_resolution = angular_resolution_degrees * pi / 180.0

    rospy.loginfo("Obstacle detection height threshold: %.3f m", altura)
    rospy.loginfo("Obstacle/free-zone detection radius: %.3f m", radio)
    rospy.loginfo("Minimum detection radius: %.3f m", minimum_detection_radius)
    rospy.loginfo("Free-zone angular resolution: %.3f deg", angular_resolution_degrees)

    point_cloud_topic = "/blue/velodyne_points"

    # Topic to publish obstacles
    pub_obstacles = rospy.Publisher("/obstacles", PointCloud2, queue_size=10)
    pub_freezone = rospy.Publisher("/free_zone", PointCloud2, queue_size=10)

    # Point cloud subscriber to filter and publish obstacles
    rospy.Subscriber(point_cloud_topic, PointCloud2, point_cloud_callback)

    # Loop to keep the node running
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
