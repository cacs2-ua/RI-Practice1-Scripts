#!/usr/bin/env python3

# Intelligent Robotics - Master's Degree in Artificial Intelligence - University of Alicante
#
# ROS node that detects obstacles from the BLUE robot Velodyne point cloud.
# It publishes:
#   - /obstacles: obstacle points projected to a fixed height.
#   - /free_zone: a circular ring of locally free navigation candidates.

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import numpy as np


# Publisher definition.
# These publishers are initialized in main() and used by publish_topics().

# Publisher definition
pub_obstacles = None
pub_freezone = None


# Obstacle detection configuration.
# altura is the minimum Z value, expressed in the Velodyne frame, from which a
# point is considered part of an obstacle. Since the LiDAR is mounted above the
# floor, floor points usually have negative Z values.
#
# radio is the local planning radius. The free-zone ring is generated at this
# radius around the BLUE robot.

# Define the height to consider the obstacles and the radius to detect them. Both variables
# are expressed in meters.

#TODO
# Set the appropriate height and radius values to detect the obstacles surrounding the object.
# Note: Consider the LiDAR sensor is mounted on the robot at a certain height. The points below the sensor
# have a negative value

altura = -0.45
radio = 6.00

MINIMUM_VALID_RANGE = 0.45
ANGULAR_BINS = 180
ANGULAR_INFLATION_BINS = 2


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
        planar_distance = np.hypot(x, y)

        #TODO
        # Add the points higher than "altura" to the "obstacles_points" array given x,y,z.
        # Note: The points added to "obstacle_points" are projected to the "altura" value, that is to say,
        # the obstacle coordinates (x,y,z) will change to (x,y,altura)

        # Height and range obstacle filter.
        # Points above altura are treated as obstacles. Very close points are ignored
        # to avoid self-detections from the BLUE robot body.

        if z > altura and MINIMUM_VALID_RANGE <= planar_distance <= radio:

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

    # Angular occupancy grid around the robot.
    # Each angular bin represents a direction around BLUE. If an obstacle is found
    # inside the planning radius, that direction and a small angular margin are
    # marked as blocked. The remaining bins are published as the free-zone ring.

    blocked_angular_bins = set()
    angular_resolution = 2.0 * np.pi / float(ANGULAR_BINS)

    # For loop to add the points corresponding to the areas free of obstacles
    # If no object is detected within a radius, the point cloud free of obstacles is generated

    for point in pc_data:
        x, y, z = point
        planar_distance = np.hypot(x, y)

        #TODO
        # Check if the point is within the radius given the x,y distance
        if MINIMUM_VALID_RANGE <= planar_distance <= radio and z > altura:

            # To create the radius of free obstacles, we need to know the angle of each point given its x,y coordinate.

            # Calculate the angle given its x,y coordinates (arcotangente)
            ang = np.arctan2(y, x)

            if ang < 0.0:
                ang += 2.0 * np.pi

            bin_index = int(ang / angular_resolution)

            for bin_offset in range(-ANGULAR_INFLATION_BINS, ANGULAR_INFLATION_BINS + 1):
                blocked_angular_bins.add((bin_index + bin_offset) % ANGULAR_BINS)

    # Free-zone ring generation.
    # A point is generated at the selected radius for each non-blocked direction.
    # These are the green candidate points used by the local planner.

    for bin_index in range(ANGULAR_BINS):
        if bin_index in blocked_angular_bins:
            continue

        ang = (bin_index + 0.5) * angular_resolution

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

    global pub_obstacles, pub_freezone, altura, radio

    # Runtime parameters.
    # These parameters allow tuning from rosrun/roslaunch without editing the file.

    altura = rospy.get_param("~obstacle_height_threshold", altura)
    radio = rospy.get_param("~free_zone_radius", radio)

    rospy.loginfo("Obstacle height threshold: %.3f m", altura)
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
