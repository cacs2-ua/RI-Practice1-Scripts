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
pub_freezone  = None

# Define the height to consider the obstacles and the radius to detect them. Both variables
# are expressed in meters.

#TODO 
# Set the appropriate height and radius values to detect the obstacles surrounding the object. 
# Note: Consider the LiDAR sensor is mounted on the robot at a certain height. The points below the sensor 
# have a negative value

# -------------------------------------------------------------------------
# BLOCK 1: Obstacle detection parameters.
# The Velodyne frame is mounted above the floor, so floor points usually have
# negative z values. Points with z greater than this threshold are considered
# possible obstacles.
# -------------------------------------------------------------------------

altura = -0.45
radio = 5.0

# Minimum distance used to avoid detecting points that belong to the robot itself.
min_obstacle_detection_distance = 0.35

# Maximum distance used to ignore very distant points that are not useful for local planning.
max_obstacle_detection_distance = 8.0


def filter_obstacles_function(point_cloud_in, altura):

    # This function receives a PointCloud2 ROS message as input and outputs a point cloud
    # containing the detected obstacles based on a given height.

    # Convert the PointCloud2 message to an array containing the x,y,z values.
    pc_data = pc2.read_points(point_cloud_in, field_names=("x", "y", "z"), skip_nans=True)

    # Array containing the parameters of the detected obstacles but initialized as void
    obstacles_points = []

    # For loop to add the points corresponding to detected obstacles
    # An obstacle is considered based on the object height.

    # -------------------------------------------------------------------------
    # BLOCK 2: Height-based obstacle filtering.
    # Points higher than the selected threshold are projected to the same z level
    # so that the planner can work in a 2D local navigation plane.
    # -------------------------------------------------------------------------

    for point in pc_data:

        x, y, z = point

        #TODO
        # Add the points higher than "altura" to the "obstacles_points" array given x,y,z. 
        # Note: The points added to "obstacle_points" are projected to the "altura" value, that is to say,
        # the obstacle coordinates (x,y,z) will change to (x,y,altura)

        distance_to_robot = np.sqrt(x * x + y * y)

        if z > altura and min_obstacle_detection_distance <= distance_to_robot <= max_obstacle_detection_distance:
            
            # Add the point to the detected obstacles:
            obstacles_points.append([x, y, altura])

    return obstacles_points


def free_zone_function(point_cloud_in, radio, altura):

    # This function receives as input a PointCloud2 ROS message and outputs a point cloud with 
    # a given radius and height. This point cloud represents the area free of obstacles

    # Convert the PointCloud2 message to an array with x,y,z values
    pc_data = pc2.read_points(point_cloud_in, field_names=("x", "y", "z"), skip_nans=True)
    
    # Array containing the parameters of the point cloud representing the area free of obstacles
    free_zone = []

    # For loop to add the points corresponding to the areas free of obstacles
    # If no object is detected within a radius, the point cloud free of obstacles is generated

    # -------------------------------------------------------------------------
    # BLOCK 3: Free-zone ring generation.
    # For each valid LiDAR direction, a point is projected onto a circular ring.
    # This ring is later used by the Naive-Valley-Path planner to choose local
    # target candidates around the robot.
    # -------------------------------------------------------------------------

    for point in pc_data:
        x, y, z = point
        
        #TODO
        # Check if the point is within the radius given the x,y distance

        distance_to_robot = np.sqrt(x * x + y * y)

        if min_obstacle_detection_distance <= distance_to_robot <= radio:

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
    global altura, radio, min_obstacle_detection_distance, max_obstacle_detection_distance

    # -------------------------------------------------------------------------
    # BLOCK 4: Runtime ROS parameters.
    # These parameters allow you to tune the detector without editing the file.
    # -------------------------------------------------------------------------

    altura = rospy.get_param("~obstacle_height_threshold", altura)
    radio = rospy.get_param("~free_zone_radius", radio)
    min_obstacle_detection_distance = rospy.get_param(
        "~min_obstacle_detection_distance",
        min_obstacle_detection_distance
    )
    max_obstacle_detection_distance = rospy.get_param(
        "~max_obstacle_detection_distance",
        max_obstacle_detection_distance
    )

    rospy.loginfo("Obstacle height threshold: %.3f m", altura)
    rospy.loginfo("Free-zone radius: %.3f m", radio)
    rospy.loginfo("Minimum obstacle detection distance: %.3f m", min_obstacle_detection_distance)
    rospy.loginfo("Maximum obstacle detection distance: %.3f m", max_obstacle_detection_distance)

    point_cloud_topic = "/blue/velodyne_points"

    # Topic to publish obstacles
    pub_obstacles = rospy.Publisher("/obstacles", PointCloud2, queue_size=10)
    pub_freezone  = rospy.Publisher("/free_zone", PointCloud2, queue_size=10)

    # Point cloud subscriber to filter and publish obstacles
    rospy.Subscriber(point_cloud_topic, PointCloud2, point_cloud_callback)

    # Loop to keep the node running
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass