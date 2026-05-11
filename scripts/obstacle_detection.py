#!/usr/bin/env python

# This node (implemented in Python) detects obstacles in the scene using the point cloud from a LiDAR sensor.
# The obstacles can be filtered using the height in Z axis given the XYZ coordinates of each point of the cloud.

# A template with "TODO" statements is provided to ease the node implementation.

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import numpy as np

# Publisher definition
pub_obstacles = None
pub_freezone = None

# Define the height to consider the obstacles and the radius to detect them. Both variables
# are expressed in meters.

#TODO
# Set the appropriate height and radius values to detect the obstacles surrounding the object.
# Note: Consider the LiDAR sensor is mounted on the robot at a certain height. The points below the sensor
# have a negative value

# -------------------------------------------------------------------------
# BLOCK 1: Obstacle detection parameters.
# The Velodyne frame is mounted above the floor, so floor points usually have
# negative z values. A threshold around -0.55 m keeps obstacle surfaces while
# rejecting most floor points. The radius limits detection to the useful local
# navigation area around BLUE.
# -------------------------------------------------------------------------
altura = -0.55
radio = 3.20


def filter_obstacles_function(point_cloud_in, altura):
    # This function receives a PointCloud2 ROS message as input and outputs a point cloud
    # containing the detected obstacles based on a given height.

    # -------------------------------------------------------------------------
    # BLOCK 2: PointCloud2 reading.
    # The raw Velodyne cloud is converted into iterable x, y, z points.
    # -------------------------------------------------------------------------

    # Convert the PointCloud2 message to an array containing the x,y,z values.
    pc_data = pc2.read_points(point_cloud_in, field_names=("x", "y", "z"), skip_nans=True)

    # Array containing the parameters of the detected obstacles but initialized as void
    obstacles_points = []

    # For loop to add the points corresponding to detected obstacles
    # An obstacle is considered based on the object height.

    for point in pc_data:
        x, y, z = point

        #TODO
        # Add the points higher than "altura" to the "obstacles_points" array given x,y,z.
        # Note: The points added to "obstacle_points" are projected to the "altura" value, that is to say,
        # the obstacle coordinates (x,y,z) will change to (x,y,altura)

        # -------------------------------------------------------------------------
        # BLOCK 3: Obstacle height and radius filtering.
        # A point is considered an obstacle if it is higher than the floor threshold
        # and inside the local navigation radius.
        # -------------------------------------------------------------------------

        horizontal_distance = np.sqrt(x * x + y * y)

        if z > altura and horizontal_distance <= radio and horizontal_distance > 0.20:
            # Add the point to the detected obstacles:
            obstacles_points.append([x, y, altura])

    return obstacles_points


def free_zone_function(point_cloud_in, radio, altura):
    # This function receives as input a PointCloud2 ROS message and outputs a point cloud with
    # a given radius and height. This point cloud represents the area free of obstacles

    # -------------------------------------------------------------------------
    # BLOCK 4: PointCloud2 reading for the free-zone ring.
    # The ring is generated from the observed LiDAR azimuths.
    # -------------------------------------------------------------------------

    # Convert the PointCloud2 message to an array with x,y,z values
    pc_data = pc2.read_points(point_cloud_in, field_names=("x", "y", "z"), skip_nans=True)

    # Array containing the parameters of the point cloud representing the area free of obstacles
    free_zone = []

    # For loop to add the points corresponding to the areas free of obstacles
    # If no object is detected within a radius, the point cloud free of obstacles is generated

    for point in pc_data:
        x, y, z = point

        #TODO
        # Check if the point is within the radius given the x,y distance

        # -------------------------------------------------------------------------
        # BLOCK 5: Free-zone ring generation.
        # For every valid LiDAR direction, a point is projected to a fixed radius.
        # This creates the green ring used by the local planner.
        # -------------------------------------------------------------------------

        horizontal_distance = np.sqrt(x * x + y * y)

        if horizontal_distance <= radio and horizontal_distance > 0.20:
            # To create the radius of free obstacles, we need to know the angle of each point given its x,y coordinate.

            # Calculate the angle given its x,y coordinates (arcotangente)
            ang = np.arctan2(y, x)

            # Calculate the new x,y coordinates given the angle and the radius
            new_x = radio * np.cos(ang)
            new_y = radio * np.sin(ang)

            # Add point to the ring with z value equal to the height
            free_zone.append([new_x, new_y, altura])

    return free_zone


def publish_topics(obstacles_points, free_zone_points):
    # This function publishes the topics related to the obstacles and free areas

    # Definition of global variables
    global pub_obstacles, pub_freezone

    # -------------------------------------------------------------------------
    # BLOCK 6: PointCloud2 publication.
    # Both outputs use the Velodyne frame because the planner works in the robot
    # local LiDAR frame.
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # BLOCK 7: Obstacle and free-zone computation.
    # The callback filters obstacle points and publishes the free-zone ring.
    # -------------------------------------------------------------------------

    # Point cloud of detected objects created by the "filter_obstacles_function" function.
    obstacles_points = filter_obstacles_function(msg, altura)

    # Point cloud of free obstacles created by the "free_zone_function" function.
    free_zone_points = free_zone_function(msg, radio, altura)

    # Publish obstacles and free areas
    publish_topics(obstacles_points, free_zone_points)


def main():
    rospy.init_node('point_cloud_filter_node', anonymous=True)

    global pub_obstacles, pub_freezone, altura, radio

    # -------------------------------------------------------------------------
    # BLOCK 8: Runtime parameters.
    # These allow tuning from rosrun without editing this file.
    # -------------------------------------------------------------------------
    altura = rospy.get_param("~obstacle_height_threshold", altura)
    radio = rospy.get_param("~free_zone_radius", radio)

    rospy.loginfo("Obstacle detection height threshold: %.3f m", altura)
    rospy.loginfo("Free-zone radius: %.3f m", radio)

    point_cloud_topic = rospy.get_param("~point_cloud_topic", "/blue/velodyne_points")

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
