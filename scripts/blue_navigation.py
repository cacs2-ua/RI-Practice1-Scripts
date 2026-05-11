#!/usr/bin/env python
#
# ROS node to calculate the trajectory towards the target while avoiding obstacles.

from __future__ import print_function

import sys
import copy
import rospy
import std_msgs.msg
import ackermann_msgs.msg
import geometry_msgs.msg
from nav_msgs.msg import Odometry
import sensor_msgs.msg
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs.point_cloud2 as pc2
import tf_conversions
import numpy as np
from math import pi, cos, sin, fabs, sqrt, atan2, tan

# Vehicle specifications
MAX_STEER_ANGLE = 24.0 * pi / 180.0  # Radians
MAX_SPEED = 1.3    # m/s
MIN_SPEED = 0.6
VEHICLE_LENGHT = 1.05


def normalize_angle(angle):
    # -------------------------------------------------------------------------
    # BLOCK 1: Angle normalization.
    # Keeps angular errors inside [-pi, pi].
    # -------------------------------------------------------------------------

    while angle > pi:
        angle -= 2.0 * pi

    while angle < -pi:
        angle += 2.0 * pi

    return angle


def clamp(value, minimum_value, maximum_value):
    # -------------------------------------------------------------------------
    # BLOCK 2: Numeric clamping helper.
    # Used to keep local targets and tuning values inside safe intervals.
    # -------------------------------------------------------------------------

    return max(minimum_value, min(maximum_value, value))


def make_point(x_value=0.0, y_value=0.0, z_value=0.0):
    # -------------------------------------------------------------------------
    # BLOCK 3: Descriptive Point constructor.
    # Avoids relying on positional ROS message constructors.
    # -------------------------------------------------------------------------

    point = geometry_msgs.msg.Point()
    point.x = x_value
    point.y = y_value
    point.z = z_value
    return point


class BlueTrajectoryPlanner(object):

    def __init__(self):
        super(BlueTrajectoryPlanner, self).__init__()

        ## ROS node initialization
        rospy.init_node("blue_planner_node", anonymous=True)

        # -------------------------------------------------------------------------
        # BLOCK 4: Base planner parameters.
        # These parameters preserve the original planner structure but make the
        # trajectory sampling denser and safer for close obstacle avoidance.
        # -------------------------------------------------------------------------

        # Parameter initialization
        self.delta_angle = rospy.get_param("~delta_angle_degrees", 4.0) * pi / 180.0
        self.delta_sample = rospy.get_param("~trajectory_sample_time_step", 0.15)
        self.max_sample = rospy.get_param("~trajectory_prediction_time", 1.10)
        self.reached_distance = rospy.get_param("~reached_distance", 0.60)
        self.slow_down_distance = rospy.get_param("~slow_down_distance", 2.0)
        self.rate = rospy.get_param("~control_rate", 8)

        #TODO Define other parameters needed for the planner.

        # -------------------------------------------------------------------------
        # BLOCK 5: Close obstacle-avoidance parameters.
        # The obstacle activation distance prevents early exaggerated detours.
        # The close bypass clearance makes the robot pass close to the obstacle,
        # while the collision margin prevents physical intersection.
        # -------------------------------------------------------------------------

        self.collision_margin = rospy.get_param("~collision_margin", 0.34)
        self.obstacle_activation_distance = rospy.get_param("~obstacle_activation_distance", 2.05)
        self.obstacle_front_min_distance = rospy.get_param("~obstacle_front_min_distance", 0.25)
        self.obstacle_corridor_half_width = rospy.get_param("~obstacle_corridor_half_width", 0.72)
        self.close_bypass_clearance = rospy.get_param("~close_bypass_clearance", 0.42)
        self.local_goal_lookahead = rospy.get_param("~local_goal_lookahead", 2.00)
        self.max_lateral_bypass = rospy.get_param("~max_lateral_bypass", 0.95)
        self.allow_reverse = rospy.get_param("~allow_reverse", False)

        self.use_camera_localization = rospy.get_param("~use_camera_localization", False)
        self.use_ground_truth_fallback = rospy.get_param("~use_ground_truth_fallback", True)
        self.wait_for_ur5_signal = rospy.get_param("~wait_for_ur5_signal", False)
        self.camera_robot_pose_index = rospy.get_param("~camera_robot_pose_index", 1)
        self.minimum_camera_motion_for_heading = rospy.get_param("~minimum_camera_motion_for_heading", 0.04)

        # -------------------------------------------------------------------------
        # BLOCK 6: Runtime state variables.
        # These variables store localization, obstacle, target, and coordination
        # state for the planner.
        # -------------------------------------------------------------------------

        # Variable initialization
        self.position = None
        self.theta = 0.0
        self.previous_camera_position = None
        self.last_camera_update_time = None

        self.obstacles = []
        self.limits = None
        self.goal_reached = False  # Wait until UR5 tells us to approach.

        #TODO Define other necessary variables.
        self.navigation_enabled = not self.wait_for_ur5_signal
        self.control_local_target = None
        self.last_bypass_side = 0

        #TODO Target position initialization. It is possible to consider several target points to maneuver and approach the UR5 robot.

        # -------------------------------------------------------------------------
        # BLOCK 7: Target point initialization.
        # The default goal is close to the UR5/platform area used in your current
        # tests. It can be changed from the terminal with _goal_x and _goal_y.
        # -------------------------------------------------------------------------

        self.goal_points = self.load_goal_points_from_ros_parameters()
        self.current_goal_index = 0
        self.goal = self.goal_points[self.current_goal_index]

        # Local path initialization to avoid errors until the first point is calculated.
        self.local_path = [
            make_point(self.local_goal_lookahead, 0.0, 0.0),
            make_point(self.local_goal_lookahead, 0.0, 0.0),
            make_point(self.local_goal_lookahead, 0.0, 0.0),
            make_point(self.local_goal_lookahead, 0.0, 0.0)
        ]

        print("Goal x: {}, y: {}".format(self.goal.x, self.goal.y))

        # TODO consider more subscribers/publishers if needed

        # -------------------------------------------------------------------------
        # BLOCK 8: Subscribers.
        # Ground truth can be used for Part 2 testing. Camera localization can be
        # enabled with _use_camera_localization:=true for the localization section.
        # -------------------------------------------------------------------------

        # Subscribers definition
        if self.use_ground_truth_fallback or not self.use_camera_localization:
            self.position_subscriber = rospy.Subscriber(
                "/blue/ground_truth",
                Odometry,
                self.position_callback,
                queue_size=1
            )

        if self.use_camera_localization:
            self.camera_position_subscriber = rospy.Subscriber(
                "/pose_array",
                geometry_msgs.msg.PoseArray,
                self.camera_pose_array_callback,
                queue_size=1
            )

        self.obstacles_subscriber = rospy.Subscriber(
            "/obstacles",
            sensor_msgs.msg.PointCloud2,
            self.obstacles_callback,
            queue_size=1
        )

        self.limits_subscriber = rospy.Subscriber(
            "/free_zone",
            sensor_msgs.msg.PointCloud2,
            self.limits_callback,
            queue_size=1
        )

        if self.wait_for_ur5_signal:
            self.start_signal_subscriber = rospy.Subscriber(
                "/ur5_object_grasped",
                std_msgs.msg.Bool,
                self.start_signal_callback,
                queue_size=1
            )

        ## Publishers definition
        self.ackermann_command_publisher = rospy.Publisher(
            "/blue/ackermann_cmd",
            ackermann_msgs.msg.AckermannDrive,
            queue_size=10,
        )

        self.marker_publisher = rospy.Publisher(
            "/local_path",
            MarkerArray,
            queue_size=10,
        )

        self.navigation_finished_publisher = rospy.Publisher(
            "/blue_navigation_finished",
            std_msgs.msg.Bool,
            queue_size=10,
        )

        rospy.loginfo("Close obstacle planner configured:")
        rospy.loginfo("  collision_margin=%.3f", self.collision_margin)
        rospy.loginfo("  obstacle_activation_distance=%.3f", self.obstacle_activation_distance)
        rospy.loginfo("  close_bypass_clearance=%.3f", self.close_bypass_clearance)
        rospy.loginfo("  allow_reverse=%s", str(self.allow_reverse))
        rospy.loginfo("  use_camera_localization=%s", str(self.use_camera_localization))

    def load_goal_points_from_ros_parameters(self):
        # -------------------------------------------------------------------------
        # BLOCK 9: Goal loading.
        # Supports either a single goal (_goal_x, _goal_y) or a waypoint list using
        # _goal_points:="x1,y1;x2,y2".
        # -------------------------------------------------------------------------

        default_goal_x = rospy.get_param("~goal_x", 0.25)
        default_goal_y = rospy.get_param("~goal_y", -0.25)

        raw_goal_points = rospy.get_param("~goal_points", "")
        parsed_goal_points = []

        if isinstance(raw_goal_points, str) and len(raw_goal_points.strip()) > 0:
            waypoint_strings = raw_goal_points.split(";")

            for waypoint_string in waypoint_strings:
                coordinates = waypoint_string.split(",")

                if len(coordinates) != 2:
                    continue

                try:
                    waypoint_x = float(coordinates[0].strip())
                    waypoint_y = float(coordinates[1].strip())
                    parsed_goal_points.append(make_point(waypoint_x, waypoint_y, 0.0))
                except ValueError:
                    rospy.logwarn("Ignoring invalid waypoint: %s", waypoint_string)

        elif isinstance(raw_goal_points, list):
            for waypoint in raw_goal_points:
                try:
                    parsed_goal_points.append(make_point(float(waypoint[0]), float(waypoint[1]), 0.0))
                except Exception:
                    rospy.logwarn("Ignoring invalid waypoint from list: %s", str(waypoint))

        if len(parsed_goal_points) == 0:
            parsed_goal_points.append(make_point(default_goal_x, default_goal_y, 0.0))

        return parsed_goal_points

    # Callbacks

    def start_signal_callback(self, signal_message):
        # -------------------------------------------------------------------------
        # BLOCK 10: Optional Part 3 start signal.
        # If enabled, BLUE waits until the UR5 reports that the object was grasped.
        # -------------------------------------------------------------------------

        if signal_message.data:
            self.navigation_enabled = True
            rospy.loginfo("Received UR5 start signal. BLUE navigation enabled.")

    #TODO modify this callback to not depend on the position given by Gazebo.
    def position_callback(self, ground_truth):
        # -------------------------------------------------------------------------
        # BLOCK 11: Ground-truth localization fallback.
        # This keeps Part 2 stable. For the localization rubric, enable the camera
        # callback using _use_camera_localization:=true.
        # -------------------------------------------------------------------------

        if self.use_camera_localization and self.last_camera_update_time is not None:
            camera_age = rospy.Time.now().to_sec() - self.last_camera_update_time

            if camera_age < 1.0:
                return

        self.position = ground_truth.pose.pose.position

        quaternion = ground_truth.pose.pose.orientation
        euler = tf_conversions.transformations.euler_from_quaternion([
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w
        ])

        self.theta = euler[2]

    def camera_pose_array_callback(self, pose_array):
        # -------------------------------------------------------------------------
        # BLOCK 12: Camera-based robot localization approximation.
        # The external camera provides the robot position in /pose_array. The robot
        # orientation is approximated from consecutive camera positions.
        # -------------------------------------------------------------------------

        if len(pose_array.poses) <= self.camera_robot_pose_index:
            return

        detected_robot_position = pose_array.poses[self.camera_robot_pose_index].position

        if detected_robot_position.z < 0.0:
            return

        current_camera_position = make_point(
            detected_robot_position.x,
            detected_robot_position.y,
            0.0
        )

        if self.previous_camera_position is not None:
            delta_x = current_camera_position.x - self.previous_camera_position.x
            delta_y = current_camera_position.y - self.previous_camera_position.y
            displacement = sqrt(delta_x * delta_x + delta_y * delta_y)

            if displacement > self.minimum_camera_motion_for_heading:
                self.theta = atan2(delta_y, delta_x)

        self.previous_camera_position = copy.deepcopy(current_camera_position)
        self.position = current_camera_position
        self.last_camera_update_time = rospy.Time.now().to_sec()

    def obstacles_callback(self, obstacles):
        # -------------------------------------------------------------------------
        # BLOCK 13: Obstacle point-cloud conversion.
        # The /obstacles topic is already expressed in the local Velodyne frame.
        # -------------------------------------------------------------------------

        pc_obstacles = pc2.read_points(obstacles, field_names=("x", "y", "z"), skip_nans=True)

        # Save as geometry_msg.Point
        self.obstacles = []

        for point in pc_obstacles:
            x, y, z = point

            if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
                continue

            new_point = make_point(x, y, z)
            self.obstacles.append(new_point)

    def limits_callback(self, limits):
        # -------------------------------------------------------------------------
        # BLOCK 14: Free-zone point-cloud conversion.
        # The free-zone ring is kept for visualization and compatibility with the
        # original navigation pipeline.
        # -------------------------------------------------------------------------

        pc_limits = pc2.read_points(limits, field_names=("x", "y", "z"), skip_nans=True)

        # Save as geometry_msg.Point
        self.limits = []

        downsample_step = rospy.get_param("~free_zone_downsample_step", 6)

        for index, point in enumerate(pc_limits):
            if index % downsample_step != 0:
                continue

            x, y, z = point

            if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
                continue

            new_point = make_point(x, y, z)
            self.limits.append(new_point)

    # Trajectory planning towards the target while avoiding obstacles. It is executed at a lower frequency.
    ''' Code implemented from the following research article:
            OpenStreetMap-Based Autonomous Navigation With LiDAR Naive-Valley-Path Obstacle Avoidance
            Miguel Ángel Muñoz Bañón, Edison Velasco Sánchez, Francisco A. Candelas, Fernando Torres
            IEEE Transactions on Intelligent Transportation Systems, 2022
    '''
    def localGoalCalculation(self):
        # -------------------------------------------------------------------------
        # BLOCK 15: Local target calculation.
        # Direct tracking is used if no close frontal obstacle exists. A close
        # bypass target is generated only when an obstacle is near the robot path.
        # This avoids exaggerated outside detours.
        # -------------------------------------------------------------------------

        if self.position is None or self.goal_reached:
            return

        if not self.navigation_enabled:
            return

        goal_in_local_axis = self.global2local(self.goal)
        relevant_obstacles = self.get_relevant_obstacles_ahead()

        if len(relevant_obstacles) == 0:
            self.last_bypass_side = 0
            local_goal = self.limit_local_target_distance(goal_in_local_axis, self.local_goal_lookahead)
        else:
            local_goal = self.calculate_close_bypass_local_goal(goal_in_local_axis, relevant_obstacles)

        self.control_local_target = copy.deepcopy(local_goal)

        self.local_path = [
            make_point(local_goal.x * 0.25, local_goal.y * 0.25, 0.0),
            make_point(local_goal.x * 0.50, local_goal.y * 0.50, 0.0),
            make_point(local_goal.x * 0.75, local_goal.y * 0.75, 0.0),
            make_point(local_goal.x, local_goal.y, 0.0)
        ]

        self.publish_local_path_markers()

    def get_relevant_obstacles_ahead(self):
        # -------------------------------------------------------------------------
        # BLOCK 16: Near-obstacle activation gate.
        # Only obstacles in front of the robot and inside a narrow corridor can
        # activate avoidance. This is the key to avoiding too early/too wide turns.
        # -------------------------------------------------------------------------

        relevant_obstacles = []

        for obstacle_point in self.obstacles:
            if obstacle_point.x < self.obstacle_front_min_distance:
                continue

            if obstacle_point.x > self.obstacle_activation_distance:
                continue

            if abs(obstacle_point.y) > self.obstacle_corridor_half_width:
                continue

            relevant_obstacles.append(obstacle_point)

        return relevant_obstacles

    def limit_local_target_distance(self, local_target, maximum_distance):
        # -------------------------------------------------------------------------
        # BLOCK 17: Local target distance limiter.
        # The control horizon is short, so the local target should not be extremely
        # far away from the robot.
        # -------------------------------------------------------------------------

        limited_target = make_point(local_target.x, local_target.y, 0.0)
        target_distance = sqrt(limited_target.x * limited_target.x + limited_target.y * limited_target.y)

        if target_distance > maximum_distance and target_distance > 0.001:
            scale = maximum_distance / target_distance
            limited_target.x *= scale
            limited_target.y *= scale

        return limited_target

    def calculate_close_bypass_local_goal(self, goal_in_local_axis, relevant_obstacles):
        # -------------------------------------------------------------------------
        # BLOCK 18: Close bypass target.
        # The robot chooses a side and places the local target just outside the
        # obstacle edge plus a small clearance. This makes it border the obstacle
        # closely instead of going far outside.
        # -------------------------------------------------------------------------

        obstacle_x_values = [point.x for point in relevant_obstacles]
        obstacle_y_values = [point.y for point in relevant_obstacles]

        obstacle_front_edge_x = max(obstacle_x_values)
        obstacle_min_y = min(obstacle_y_values)
        obstacle_max_y = max(obstacle_y_values)
        obstacle_center_y = float(np.median(obstacle_y_values))

        if self.last_bypass_side != 0:
            bypass_side = self.last_bypass_side
        elif abs(goal_in_local_axis.y) > 0.15:
            bypass_side = 1 if goal_in_local_axis.y > 0.0 else -1
        else:
            bypass_side = -1 if obstacle_center_y >= 0.0 else 1

        self.last_bypass_side = bypass_side

        if bypass_side > 0:
            target_y = obstacle_max_y + self.close_bypass_clearance
        else:
            target_y = obstacle_min_y - self.close_bypass_clearance

        target_y = clamp(target_y, -self.max_lateral_bypass, self.max_lateral_bypass)

        target_x = obstacle_front_edge_x + 0.45
        target_x = clamp(target_x, 0.85, self.local_goal_lookahead)

        bypass_target = make_point(target_x, target_y, 0.0)

        rospy.loginfo_throttle(
            1.0,
            "Close bypass target: x=%.3f y=%.3f side=%d relevant_obstacles=%d",
            bypass_target.x,
            bypass_target.y,
            bypass_side,
            len(relevant_obstacles)
        )

        return bypass_target

    def publish_local_path_markers(self):
        # -------------------------------------------------------------------------
        # BLOCK 19: Local target visualization.
        # Publishes the selected local path in RViz using the blue/velodyne frame.
        # -------------------------------------------------------------------------

        marker_msg = MarkerArray()

        for marker_id, point in enumerate(self.local_path):
            marker = Marker()
            marker.header.frame_id = "blue/velodyne"
            marker.header.stamp = rospy.Time.now()
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = point
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.35
            marker.scale.y = 0.35
            marker.scale.z = 0.35
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker_msg.markers.append(marker)

        self.marker_publisher.publish(marker_msg)

    # Calculate the control commands to reach the planned local target.
    def controlActionCalculation(self):
        # Detect if the trajectory is finished.
        if self.position is None or self.goal_reached:
            return

        ackermann_control = ackermann_msgs.msg.AckermannDrive()
        ackermann_control.speed, ackermann_control.steering_angle = 0.0, 0.0

        if not self.navigation_enabled:
            self.ackermann_command_publisher.publish(ackermann_control)
            return

        # Detect if the robot has reached the following target point.
        if self.distance(self.position, self.goal) < self.reached_distance:
            if self.current_goal_index + 1 < len(self.goal_points):
                self.current_goal_index += 1
                self.goal = self.goal_points[self.current_goal_index]
                rospy.loginfo("Switching to next goal x=%.3f y=%.3f", self.goal.x, self.goal.y)
                return

            print("Goal reached")
            self.goal_reached = True
            self.navigation_finished_publisher.publish(std_msgs.msg.Bool(data=True))
            self.ackermann_command_publisher.publish(ackermann_control)
            return

        # Reduce the velocity when the robot is reaching the target.
        goal_distance = self.distance(self.position, self.goal)

        if goal_distance - self.reached_distance < self.slow_down_distance:
            speed = MIN_SPEED
            self.control_local_target = self.global2local(self.goal)
        else:
            speed = MAX_SPEED

        if self.control_local_target is None:
            self.control_local_target = self.limit_local_target_distance(
                self.global2local(self.goal),
                self.local_goal_lookahead
            )

        local_target = self.control_local_target

        # Variable initialization
        min_error = float("inf")
        best_command_found = False

        # Check possible action control commands
        for steer in np.arange(-MAX_STEER_ANGLE, MAX_STEER_ANGLE + 0.01, self.delta_angle):
            if abs(steer) < 0.01:
                steer = 0.0

            # Reduce the velocity when the turn is big.
            k_sp = (MAX_STEER_ANGLE - abs(steer)) / MAX_STEER_ANGLE
            speed2 = max(speed * k_sp, MIN_SPEED)

            # Check forwards and backwards movement.
            if self.allow_reverse:
                directions = [-speed2, speed2]
            else:
                directions = [speed2]

            for direction_speed in directions:
                flag_collision_risk = False
                trajectory_points = []
                final_yaw = 0.0

                #TODO Calculate turn radius and velocity using the robot kinematics.

                # -------------------------------------------------------------------------
                # BLOCK 20: Bicycle-model angular velocity.
                # R = L / tan(delta)
                # omega = v / R
                # -------------------------------------------------------------------------

                if abs(steer) < 0.0001:
                    angular_velocity = 0.0
                else:
                    turn_radius = VEHICLE_LENGHT / tan(steer)
                    angular_velocity = direction_speed / turn_radius

                # Calculate the trajectory using the angle rotation. Several points are sampled from the trajectory over time,
                # therefore the sample variable is equivalent to the t variable in the robot kinematic equation.
                for sample in np.arange(self.delta_sample, self.max_sample + 0.01, self.delta_sample):
                    local_point = geometry_msgs.msg.Point()

                    #TODO calculate trajectory points using the robot kinematics.

                    # -------------------------------------------------------------------------
                    # BLOCK 21: Trajectory point generation using the assignment equations.
                    # The robot is assumed to start at local pose (0, 0, 0).
                    # x = v * cos(omega * t) * t
                    # y = v * sin(omega * t) * t
                    # -------------------------------------------------------------------------

                    local_yaw = angular_velocity * sample
                    local_point.x = direction_speed * cos(local_yaw) * sample
                    local_point.y = direction_speed * sin(local_yaw) * sample
                    local_point.z = 0.0

                    trajectory_points.append(copy.deepcopy(local_point))
                    final_yaw = local_yaw

                    #TODO Detect collisions risk.

                    # -------------------------------------------------------------------------
                    # BLOCK 22: Collision-risk detection.
                    # A trajectory is rejected if any sampled point enters the
                    # configured obstacle safety radius.
                    # -------------------------------------------------------------------------

                    for obstacle in self.obstacles:
                        if obstacle.x < -0.20:
                            continue

                        if self.distance(local_point, obstacle) < self.collision_margin:
                            flag_collision_risk = True
                            break

                    if flag_collision_risk:
                        break

                # If there is no collision, evaluate the trajectory.
                if not flag_collision_risk and len(trajectory_points) > 0:
                    #TODO estimate the trajectory evaluation in terms of the distance and orientation error to the local target (self.local_path[-2]).

                    # -------------------------------------------------------------------------
                    # BLOCK 23: Trajectory score.
                    # The score combines final distance to the local target, heading
                    # alignment, steering effort, and a reverse penalty.
                    # -------------------------------------------------------------------------

                    final_point = trajectory_points[-1]
                    error = self.calculate_trajectory_score(
                        final_point,
                        final_yaw,
                        local_target,
                        steer,
                        direction_speed
                    )

                    if error < min_error:
                        min_error = error
                        best_command_found = True
                        ackermann_control.steering_angle = steer
                        ackermann_control.speed = direction_speed

        if not best_command_found:
            # -------------------------------------------------------------------------
            # BLOCK 24: Conservative fallback.
            # If no sampled trajectory is safe, the robot stops instead of forcing a
            # risky command. This prevents collision with the obstacle.
            # -------------------------------------------------------------------------

            rospy.logwarn_throttle(1.0, "No collision-free trajectory found. Stopping BLUE.")
            ackermann_control.speed = 0.0
            ackermann_control.steering_angle = 0.0

        # Publish message
        self.ackermann_command_publisher.publish(ackermann_control)

    def calculate_trajectory_score(self, final_point, final_yaw, local_target, steering_angle, direction_speed):
        # -------------------------------------------------------------------------
        # BLOCK 25: Score calculation for trajectory selection.
        # Lower score means better trajectory.
        # -------------------------------------------------------------------------

        distance_error = self.distance(final_point, local_target)

        desired_heading = atan2(
            local_target.y - final_point.y,
            local_target.x - final_point.x
        )

        orientation_error = abs(normalize_angle(desired_heading - final_yaw))

        steering_penalty = 0.12 * abs(steering_angle) / MAX_STEER_ANGLE
        reverse_penalty = 1.00 if direction_speed < 0.0 else 0.0

        low_progress_penalty = 0.0
        if local_target.x > 0.0 and final_point.x < 0.05:
            low_progress_penalty = 0.50

        score = (
            1.20 * distance_error +
            0.55 * orientation_error +
            steering_penalty +
            reverse_penalty +
            low_progress_penalty
        )

        return score

    def distance(self, p1, p2):
        return sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    # Angle from point p2 to p1
    def angle(self, p1, p2):
        return atan2((p1.y - p2.y), p1.x - p2.x)

    # Transformation from global to local position
    def global2local(self, p):
        result = geometry_msgs.msg.Point()

        if self.position is None:
            return make_point(p.x, p.y, 0.0)

        # Translation
        x = (p.x - self.position.x)
        y = (p.y - self.position.y)

        # Rotation
        result.x = x * cos(-self.theta) - y * sin(-self.theta)
        result.y = x * sin(-self.theta) + y * cos(-self.theta)
        result.z = 0.0

        return result

    # From spherical to cartesian coordinates
    def spherical2Cartesian(self, depth, azimuth):
        sin_azimuth = sin(azimuth)
        cos_azimuth = cos(azimuth)
        p = geometry_msgs.msg.Point()
        p.x = depth * cos_azimuth
        p.y = depth * sin_azimuth
        p.z = 0.0
        return p

    # From cartesian to spherical coordinates
    def cartesian2Spherical(self, x, y):
        depth = sqrt((x * x) + (y * y))

        azimuth = atan2(y, x)

        if azimuth < 0:
            azimuth += 2 * pi

        if azimuth >= 2 * pi:
            azimuth -= 2 * pi

        return depth, azimuth

    def run(self):
        # Control loop
        rate = rospy.Rate(self.rate)
        count = 0

        while not rospy.is_shutdown():
            if count == 0:
                self.localGoalCalculation()
                count = 3
            else:
                count -= 1

            self.controlActionCalculation()
            rate.sleep()


def main():
    try:
        print("Init blue_planner_node")
        node = BlueTrajectoryPlanner()
        node.run()

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
