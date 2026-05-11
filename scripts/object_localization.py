#!/usr/bin/env python3

# This node (implemented in Python) computes the position (xyz) of the objects detected by an RGBD camera.
# The centroid of each individual detected object is obtained using the RGB to HSV color filtering.
# The xyz objects coordinates (with respect to the camera) are calculated considering the camera-object distance and the insitric parameters of the camera are previously known.

# A template with "TODO" statements is provided to ease the node implementation.

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import PoseArray, Pose
from math import sqrt, pow


# Global variables.
# This block stores the publishers, centroids, depth values, and detection flags
# shared by the RGB, depth, and camera-info callbacks.

# Define global variables (publishers and flags)
filtered_obj_pub = None
filtered_blue_pub = None
pose_array_pub = None

centroide_obj = None
centroide_blue = None

depth_obj = None
depth_blue = None

obt_detec = False
robot_detec = False


def filter_img_objects(color_image, lower, upper):
    # Function to filter an RGB image to a given color in HSV range.
    # This function ouputs the filtered image, the object centroid in image coordinates, and a flag
    # indicating if an object was detected or not.

    # HSV color filtering without deleting small objects.
    # The red object is very small in the overhead camera image. Therefore, this
    # implementation avoids aggressive morphological opening operations that can
    # completely remove the red cube from the binary mask.

    # Convert the RGB image to HSV channel for the filtering
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # Filter the image to get the mask
    if isinstance(lower, (list, tuple)) and isinstance(upper, (list, tuple)):
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for lower_limit, upper_limit in zip(lower, upper):
            partial_mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
            mask = cv2.bitwise_or(mask, partial_mask)
    else:
        mask = cv2.inRange(hsv_image, lower, upper)

    # Keep the mask almost untouched because the red object is very small.
    # A small closing operation can fill tiny holes without deleting the object.
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    filtered_image = cv2.bitwise_and(color_image, color_image, mask=mask)

    # Obtain the mask contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Save the coordinates of the object centroid in pixels, and a detected/not detected flag for the object.

    centroid = None  # (px,py)
    detection = False  # detected/not detected object flag

    #TODO
    # Calculate the mask centroid
    # Obtain the contour with the biggest area (cv2.contourArea) given the contours of the filtered object.
    # Obtaing the object centroid (px,py) using the image moments (cv2.moments)

    # Centroid from the largest contour.
    # A very small minimum area is used because the red object occupies few pixels
    # in the top-view camera image.

    if contours:

        # Find the contour with the biggest area to obtain its contour
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)

        # Obtain the image moments
        minimum_valid_area = 1.0

        if largest_area >= minimum_valid_area:
            moments = cv2.moments(largest_contour)

            # Calculate the centroid coordinates
            if moments["m00"] != 0:
                centroid_x = int(moments["m10"] / moments["m00"])
                centroid_y = int(moments["m01"] / moments["m00"])
                centroid = (centroid_x, centroid_y)
                detection = True

    # Fallback: if the contour moment fails, compute the centroid directly from
    # the non-zero mask pixels. This is useful for very small red blobs.
    if not detection:
        mask_y_coordinates, mask_x_coordinates = np.nonzero(mask)

        if len(mask_x_coordinates) > 0:
            centroid_x = int(np.mean(mask_x_coordinates))
            centroid_y = int(np.mean(mask_y_coordinates))
            centroid = (centroid_x, centroid_y)
            detection = True
        else:
            # flag to indicate that there was no object in the mask
            detection = False

    return filtered_image, centroid, detection


def img_xyz(centroid, depth, camera_matrix):

    # This function transforms a pixel (centroid) from a depth image to a XYZ coordinate using the intrinsic parameters of the camera (camera_matrix).
    # This function outputs the x,y,z of the detected object.

    # Intrinsic camera parameter extraction.
    # The camera matrix K is stored as:
    # [fx, 0, cx, 0, fy, cy, 0, 0, 1]

    # TODO
    # Extract the intrinsic parameters of the camera_matrix variable given the following order:

    # fx 0  cx
    # 0  fy cy
    # 0  0  1

    fx = camera_matrix[0]
    cx = camera_matrix[2]
    fy = camera_matrix[4]
    cy = camera_matrix[5]

    # TODO
    # Calculate the x,y,z coordinates of the object given the intrinsic parameters of the camera (fx,fy,cx,cy) and
    # the object centroid (px,py).

    px, py = centroid

    z = depth
    x = (px - cx) * z / fx
    y = (py - cy) * z / fy

    return x, y, z


def color_image_callback(color_image_msg):

    # Callback function of the RGB image.
    # This function filters the RGB image and publish two messages with the filtered images of the
    # detected objects (red object and Blue robot)

    # Definition of global variables
    global filtered_obj_pub, filtered_blue_pub, centroide_obj, centroide_blue, obt_detec, robot_detec

    # RGB image conversion.
    # The ROS Image message is converted into an OpenCV BGR image.

    # Convert the image message from ROS type to OpenCV
    bridge = CvBridge()
    color_image = bridge.imgmsg_to_cv2(color_image_msg, desired_encoding="bgr8")

    # BLUE robot color segmentation.
    # This keeps the original HSV range from the base code.

    # Filter robot Blue
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([120, 255, 255])
    filtered_blue, centroide_blue, robot_detec = filter_img_objects(color_image, lower_blue, upper_blue)

    # Publish the filtered image of the robot
    filtered_image_msg = bridge.cv2_to_imgmsg(filtered_blue, encoding="bgr8")
    filtered_blue_pub.publish(filtered_image_msg)

    #TODO
    # Filter the red color objects corresponding to the objects to manipulate in the scene,
    # based on the previous example of blue color filtering.

    # Red object color segmentation.
    # Red is split into two HSV intervals because red hue wraps around the HSV axis.

    # Filter red color object (HSV limits)
    # The simulated red cube is very small and may appear dark due to Gazebo lighting.
    # Therefore, the HSV red range is intentionally broad.
    lower_red = [
        np.array([0, 20, 10]),
        np.array([140, 20, 10])
    ]
    upper_red = [
        np.array([30, 255, 255]),
        np.array([179, 255, 255])
    ]

    filtered_obj, centroide_obj, obt_detec = filter_img_objects(color_image, lower_red, upper_red)

    # Publish the filtered image of the object
    filtered_image_msg = bridge.cv2_to_imgmsg(filtered_obj, encoding="bgr8")
    filtered_obj_pub.publish(filtered_image_msg)


def get_depth_value_in_meters(depth_image, centroid):

    # Robust depth extraction.
    # A small window around the centroid is used. If the value is in millimetres,
    # it is converted to metres.

    if centroid is None:
        return None

    px, py = centroid
    image_height, image_width = depth_image.shape[:2]

    if px < 0 or px >= image_width or py < 0 or py >= image_height:
        return None

    window_radius = 2

    x_min = max(px - window_radius, 0)
    x_max = min(px + window_radius + 1, image_width)
    y_min = max(py - window_radius, 0)
    y_max = min(py + window_radius + 1, image_height)

    depth_window = depth_image[y_min:y_max, x_min:x_max].astype(np.float32)

    valid_depth_values = depth_window[np.isfinite(depth_window)]
    valid_depth_values = valid_depth_values[valid_depth_values > 0.0]

    if valid_depth_values.size == 0:
        return None

    depth_value = float(np.median(valid_depth_values))

    if depth_value > 20.0:
        depth_value = depth_value / 1000.0

    return depth_value


def depth_image_callback(depth_image_msg):

    # Callback function of the depth image
    # This callback function obtains the depth of the pixel centroid (px,py) given the depth image.
    # If the centroid does not exist because an object is not detected, the depth is not calculated.

    # Definition of global variables
    global centroide_obj, centroide_blue, depth_obj, depth_blue, obt_detec, robot_detec

    # Depth image conversion.
    # The depth image is read using passthrough encoding to preserve its original type.

    # Transform the depth image message to a OpenCV image
    bridge = CvBridge()
    depth_image = bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding="passthrough")

    # Obtain the data from the depth image given the centroid.
    # If the detected object or robot variables are False, then there is no depth.

    # TODO
    # Obtain the camera-object distance (depth) of the object and robot in the scene, given the previous description.
    # Note: The depth image data must be converted from millimeters to meters.

    # Depth computation for the object and BLUE robot.

    if obt_detec:
        depth_obj = get_depth_value_in_meters(depth_image, centroide_obj)

    if robot_detec:
        depth_blue = get_depth_value_in_meters(depth_image, centroide_blue)


def camera_info_callback(camera_info_msg):

    # Callback function of the camera information.
    # This callback obtains the intrinsic parameters of the camera.
    # Once the RGB and depth image data are obtained, the object is located with
    # respect to the camera.

    # Definition of global variables
    global centroide_obj, centroide_blue, depth_obj, depth_blue, pose_array_pub, obt_detec, robot_detec

    # XYZ localization and PoseArray publication.
    # z = -1 means that the object or robot has not been detected correctly.

    # Obtain the object localization
    # Initialize xyz variables of each object.
    # If an object is not detected in the image, z=-1, x=0, y=0

    x_obj, y_obj, z_obj = 0, 0, -1
    x_blue, y_blue, z_blue = 0, 0, -1

    if obt_detec and depth_obj is not None:
        x_obj, y_obj, z_obj = img_xyz(centroide_obj, depth_obj, camera_info_msg.K)

    if robot_detec and depth_blue is not None:
        x_blue, y_blue, z_blue = img_xyz(centroide_blue, depth_blue, camera_info_msg.K)

    # Invert y axis so that it matches the global axis.
    y_obj, y_blue = -y_obj, -y_blue

    # Print the object localization
    print("XYZ object: {:.3f}, {:.3f}, {:.3f}".format(x_obj, y_obj, z_obj))
    print("XYZ robot : {:.3f}, {:.3f}, {:.3f}".format(x_blue, y_blue, z_blue))

    # Define a PoseArray message
    pose_array_msg = PoseArray()

    # Define the message header (timestamp and frame)
    pose_array_msg.header.stamp = rospy.Time.now()
    pose_array_msg.header.frame_id = "camera_link"

    # Object pose
    pose_obj = Pose()
    pose_obj.position.x = x_obj
    pose_obj.position.y = y_obj
    pose_obj.position.z = z_obj
    pose_obj.orientation.w = 1.0

    # Robot pose
    pose_blue = Pose()
    pose_blue.position.x = x_blue
    pose_blue.position.y = y_blue
    pose_blue.position.z = z_blue
    pose_blue.orientation.w = 1.0

    # Add the object pose to the PoseArray message
    pose_array_msg.poses.append(pose_obj)

    # Add the robot pose to the PoseArray message
    pose_array_msg.poses.append(pose_blue)

    # Publish the PoseArray message
    pose_array_pub.publish(pose_array_msg)


def ros_node():
    rospy.init_node('Object_localization', anonymous=True)

    # Subscribes to the following topics:
    # - RGB image from the camera
    # - Depth image from the camera
    # - Intrinsic parameters of the camera (fx,fy,cx,cy)

    # TODO
    # Add the topics to which the program needs to be subscribed. Follow the example of how to
    # subscribe to the "/camera/color/image_raw" topic to obtaing the RGB image.
    # The subscribers to obtain the depth image and the intrinsic parameters of the camera must be implemented.

    # Publishers.
    # Publishers are created before subscribers to avoid callback race conditions.

    # global variables
    global filtered_obj_pub, filtered_blue_pub, pose_array_pub

    # Define publishers (filtered images and objects poses)
    filtered_obj_pub = rospy.Publisher('/filtered_image/object', Image, queue_size=1)
    filtered_blue_pub = rospy.Publisher('/filtered_image/robot', Image, queue_size=1)
    pose_array_pub = rospy.Publisher('/pose_array', PoseArray, queue_size=10)

    # Subscribers.
    # The default topics correspond to the simulated Realsense camera.

    color_image_topic = rospy.get_param("~color_image_topic", "/camera/color/image_raw")
    depth_image_topic = rospy.get_param("~depth_image_topic", "/camera/depth/image_raw")
    camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")

    rospy.loginfo("Subscribing to RGB image topic: %s", color_image_topic)
    rospy.loginfo("Subscribing to depth image topic: %s", depth_image_topic)
    rospy.loginfo("Subscribing to camera info topic: %s", camera_info_topic)

    color_image_sub = rospy.Subscriber(color_image_topic, Image, color_image_callback)
    depth_image_sub = rospy.Subscriber(depth_image_topic, Image, depth_image_callback)
    camera_info_sub = rospy.Subscriber(camera_info_topic, CameraInfo, camera_info_callback)

    rospy.spin()


if __name__ == '__main__':
    try:
        ros_node()
    except rospy.ROSInterruptException:
        pass
