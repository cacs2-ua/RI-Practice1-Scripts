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

# -------------------------------------------------------------------------
# BLOCK 1: LiDAR obstacle-filter configuration.
# The Velodyne frame is mounted above the floor, so the floor usually appears
# with negative z values. Points higher than altura are considered possible
# obstacle points around the BLUE robot.
# -------------------------------------------------------------------------

#TODO 
# Set the appropriate height and radius values to detect the obstacles surrounding the object. 
# Note: Consider the LiDAR sensor is mounted on the robot at a certain height. The points below the sensor 
# have a negative value

altura = -0.55
radio = 4.00

minimum_lidar_distance = 0.35


def filter_obstacles_function(point_cloud_in, altura):

    # This function receives a PointCloud2 ROS message as input and outputs a point cloud
    # containing the detected obstacles based on a given height.

    # Convert the PointCloud2 message to an array containing the x,y,z values.
    pc_data = pc2.read_points(point_cloud_in, field_names=("x", "y", "z"), skip_nans=True)

    # Array containing the parameters of the detected obstacles but initialized as void
    obstacles_points = []

    # -------------------------------------------------------------------------
    # BLOCK 2: Obstacle point selection.
    # A point is an obstacle if it is above the height threshold, inside the local
    # detection radius, and not part of the very near robot body/sensor noise.
    # The output points are projected to z = altura, as required by the template.
    # -------------------------------------------------------------------------

    # For loop to add the points corresponding to detected obstacles
    # An obstacle is considered based on the object height.

    for point in pc_data:

        x, y, z = point
        xy_distance = np.sqrt(x * x + y * y)

        #TODO
        # Add the points higher than "altura" to the "obstacles_points" array given x,y,z. 
        # Note: The points added to "obstacle_points" are projected to the "altura" value, that is to say,
        # the obstacle coordinates (x,y,z) will change to (x,y,altura)

        if z > altura and xy_distance <= radio and xy_distance >= minimum_lidar_distance: # height condition compared to Z
            
            # Add the point to the detected obstacles:
            obstacles_points.append([x, y, altura])

    return obstacles_points


def free_zone_function(point_cloud_in, radio,altura):

    # This function receives as input a PointCloud2 ROS message and outputs a point cloud with 
    # a given radius and height. This point cloud represents the area free of obstacles

    # Convert the PointCloud2 message to an array with x,y,z values
    pc_data = pc2.read_points(point_cloud_in, field_names=("x", "y", "z"), skip_nans=True)
    
    # Array containing the parameters of the point cloud representing the area free of obstacles
    free_zone = []

    # -------------------------------------------------------------------------
    # BLOCK 3: Free-zone ring generation.
    # For each valid LiDAR ray inside the radius, the point direction is projected
    # onto a ring of radius radio. The local planner uses this ring as the set of
    # candidate valley limits.
    # -------------------------------------------------------------------------

    # For loop to add the points corresponding to the areas free of obstacles
    # If no object is detected within a radius, the point cloud free of obstacles is generated

    for point in pc_data:
        x, y, z = point
        xy_distance = np.sqrt(x * x + y * y)
        
        #TODO
        # Check if the point is within the radius given the x,y distance
        if xy_distance <= radio and xy_distance >= minimum_lidar_distance:

            # To create the radius of free obstacles, we need to know the angle of each point given its x,y coordinate.
        
            # Calculate the angle given its x,y coordinates (arcotangente)
            ang = np.arctan2(y, x)

            # Calculate the new x,y coordinates given the angle and the radius
            new_x = radio * np.cos(ang)
            new_y = radio * np.sin(ang)
            
            # Add point to the ring with z value equal to the height
            free_zone.append([new_x, new_y, altura])

    return free_zone


def publish_topics(obstacles_points,free_zone_points):

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
    free_zone_points = free_zone_function(msg, radio,altura)

    # Publish obstacles and free areas
    publish_topics(obstacles_points, free_zone_points)


def main():
    rospy.init_node('point_cloud_filter_node', anonymous=True)

    global pub_obstacles, pub_freezone, altura, radio, minimum_lidar_distance

    # -------------------------------------------------------------------------
    # BLOCK 4: Runtime ROS parameters.
    # These allow tuning obstacle detection without editing the file.
    # -------------------------------------------------------------------------

    altura = rospy.get_param("~altura", altura)
    radio = rospy.get_param("~radio", radio)
    minimum_lidar_distance = rospy.get_param("~minimum_lidar_distance", minimum_lidar_distance)

    rospy.loginfo("Obstacle detection altura=%.3f m, radio=%.3f m", altura, radio)

    point_cloud_topic = rospy.get_param("~point_cloud_topic", "/blue/velodyne_points")

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