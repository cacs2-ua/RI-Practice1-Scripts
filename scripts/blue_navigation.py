#!/usr/bin/env python3
#
# Intelligent Robotics - Master's Degree in Artificial Intelligence - University of Alicante
# ROS node for robot BLUE navigation.

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
from math import pi, cos, sin, sqrt, atan2, tan


# Vehicle specifications
MAX_STEER_ANGLE = 24.0 * pi / 180.0
MAX_SPEED = 1.3
MIN_SPEED = 0.6
VEHICLE_LENGHT = 1.05


class BlueTrajectoryPlanner(object):

    def __init__(self):
        super(BlueTrajectoryPlanner, self).__init__()

        rospy.init_node("blue_planner_node", anonymous=True)

        # ---------------------------------------------------------------------
        # Basic planner parameters.
        # ---------------------------------------------------------------------

        self.delta_angle = rospy.get_param("~delta_angle", 4.0 * pi / 180.0)
        self.delta_sample = rospy.get_param("~delta_sample", 0.10)
        self.max_sample = rospy.get_param("~max_sample", 2.40)

        self.reached_distance = rospy.get_param("~reached_distance", 0.8)
        self.slow_down_distance = rospy.get_param("~slow_down_distance", 2.0)
        self.rate = rospy.get_param("~rate", 6)

        # IMPORTANT:
        # Reverse is disabled by default because the rubric asks for navigation
        # and obstacle avoidance, not repeated forward/backward recovery.
        self.allow_reverse = rospy.get_param("~allow_reverse", False)

        # ---------------------------------------------------------------------
        # Collision and local-target tuning.
        # ---------------------------------------------------------------------

        self.collision_margin = rospy.get_param("~collision_margin", 0.50)
        self.robot_self_filter_radius = rospy.get_param("~robot_self_filter_radius", 0.35)

        self.candidate_clearance = rospy.get_param("~candidate_clearance", 1.05)
        self.blocked_sector_clearance = rospy.get_param("~blocked_sector_clearance", 1.20)

        self.front_obstacle_sector_width = rospy.get_param("~front_obstacle_sector_width", 1.60)
        self.front_obstacle_detection_distance = rospy.get_param("~front_obstacle_detection_distance", 4.50)
        self.front_obstacle_slow_down_distance = rospy.get_param("~front_obstacle_slow_down_distance", 3.00)

        self.obstacle_approach_speed = rospy.get_param("~obstacle_approach_speed", 0.45)
        self.creep_speed = rospy.get_param("~creep_speed", 0.35)
        self.emergency_stop_distance = rospy.get_param("~emergency_stop_distance", 0.85)

        # ---------------------------------------------------------------------
        # Trajectory score weights.
        # ---------------------------------------------------------------------

        self.distance_error_weight = rospy.get_param("~distance_error_weight", 1.0)
        self.orientation_error_weight = rospy.get_param("~orientation_error_weight", 0.7)
        self.clearance_error_weight = rospy.get_param("~clearance_error_weight", 0.35)
        self.reverse_motion_penalty = rospy.get_param("~reverse_motion_penalty", 5.0)
        self.steering_penalty_weight = rospy.get_param("~steering_penalty_weight", 0.03)
        self.steering_change_penalty_weight = rospy.get_param("~steering_change_penalty_weight", 0.20)
        self.side_preference_weight = rospy.get_param("~side_preference_weight", 0.55)

        self.minimum_displacement_for_heading_update = rospy.get_param(
            "~minimum_displacement_for_heading_update",
            0.03
        )

        self.use_camera_localization = rospy.get_param("~use_camera_localization", True)
        self.use_ground_truth_fallback = rospy.get_param("~use_ground_truth_fallback", False)

        self.wait_for_ur5_signal = rospy.get_param("~wait_for_ur5_signal", False)
        self.navigation_enabled = not self.wait_for_ur5_signal

        # ---------------------------------------------------------------------
        # State variables.
        # ---------------------------------------------------------------------

        self.position = None
        self.theta = rospy.get_param("~initial_theta", 0.0)

        self.previous_camera_position = None
        self.camera_localization_received = False

        self.obstacles = []
        self.limits = None

        self.goal_reached = False

        self.avoidance_side_sign = 0.0
        self.last_selected_steering = 0.0
        self.last_selected_speed = 0.0

        self.target_points = self.read_target_points_from_ros_params()
        self.current_target_index = 0
        self.goal = copy.deepcopy(self.target_points[self.current_target_index])

        self.local_target = copy.deepcopy(self.goal)
        self.local_path = [copy.deepcopy(self.local_target) for _ in range(4)]

        print("Goal x: {}, y: {}".format(self.goal.x, self.goal.y))

        # ---------------------------------------------------------------------
        # Subscribers.
        # ---------------------------------------------------------------------

        if self.use_camera_localization:
            self.position_subscriber = rospy.Subscriber(
                "/pose_array",
                geometry_msgs.msg.PoseArray,
                self.position_callback,
                queue_size=1
            )
        else:
            self.position_subscriber = rospy.Subscriber(
                "/blue/ground_truth",
                Odometry,
                self.ground_truth_position_callback,
                queue_size=1
            )

        if self.use_ground_truth_fallback:
            self.ground_truth_fallback_subscriber = rospy.Subscriber(
                "/blue/ground_truth",
                Odometry,
                self.ground_truth_position_callback,
                queue_size=1
            )

        self.obstacles_subscriber = rospy.Subscriber(
            "/obstacles",
            sensor_msgs.msg.PointCloud2,
            self.obstacles_callback,
            queue_size=1
        )

        self.free_zone_subscriber = rospy.Subscriber(
            "/free_zone",
            sensor_msgs.msg.PointCloud2,
            self.limits_callback,
            queue_size=1
        )

        self.ur5_object_grasped_subscriber = rospy.Subscriber(
            "/ur5_object_grasped",
            std_msgs.msg.Bool,
            self.ur5_object_grasped_callback,
            queue_size=1
        )

        # ---------------------------------------------------------------------
        # Publishers.
        # ---------------------------------------------------------------------

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

        self.blue_goal_reached_publisher = rospy.Publisher(
            "/blue_goal_reached",
            std_msgs.msg.Bool,
            queue_size=1,
            latch=True
        )

        rospy.loginfo("Camera localization enabled: %s", str(self.use_camera_localization))
        rospy.loginfo("Ground-truth fallback enabled: %s", str(self.use_ground_truth_fallback))
        rospy.loginfo("Waiting for UR5 signal before moving: %s", str(self.wait_for_ur5_signal))
        rospy.loginfo("Allow reverse: %s", str(self.allow_reverse))
        rospy.loginfo("Collision margin: %.3f m", self.collision_margin)
        rospy.loginfo("Candidate clearance: %.3f m", self.candidate_clearance)
        rospy.loginfo("Planning horizon: %.3f s", self.max_sample)
        rospy.loginfo("Target points: %s", self.target_points_to_string())

    # -------------------------------------------------------------------------
    # Target parser.
    # -------------------------------------------------------------------------

    def read_target_points_from_ros_params(self):
        target_points = []
        target_points_string = rospy.get_param("~target_points", "")

        if target_points_string:
            raw_points = target_points_string.split(";")

            for raw_point in raw_points:
                clean_point = raw_point.strip()

                if not clean_point:
                    continue

                coordinates = clean_point.split(",")

                if len(coordinates) != 2:
                    rospy.logwarn("Invalid target point ignored: %s", clean_point)
                    continue

                target_point = geometry_msgs.msg.Point()
                target_point.x = float(coordinates[0])
                target_point.y = float(coordinates[1])
                target_point.z = 0.0
                target_points.append(target_point)

        if len(target_points) == 0:
            target_point = geometry_msgs.msg.Point()
            target_point.x = rospy.get_param("~goal_x", 5.0)
            target_point.y = rospy.get_param("~goal_y", 3.0)
            target_point.z = 0.0
            target_points.append(target_point)

        return target_points

    def target_points_to_string(self):
        points_as_text = []

        for point in self.target_points:
            points_as_text.append("({:.3f}, {:.3f})".format(point.x, point.y))

        return " -> ".join(points_as_text)

    # -------------------------------------------------------------------------
    # Localization callbacks.
    # -------------------------------------------------------------------------

    def position_callback(self, pose_array):
        if len(pose_array.poses) < 2:
            return

        detected_robot_position = pose_array.poses[1].position

        if detected_robot_position.z < 0:
            return

        new_position = geometry_msgs.msg.Point()
        new_position.x = detected_robot_position.x
        new_position.y = detected_robot_position.y
        new_position.z = 0.0

        if self.previous_camera_position is not None:
            delta_x = new_position.x - self.previous_camera_position.x
            delta_y = new_position.y - self.previous_camera_position.y
            displacement = sqrt(delta_x * delta_x + delta_y * delta_y)

            if displacement >= self.minimum_displacement_for_heading_update:
                self.theta = atan2(delta_y, delta_x)

        self.previous_camera_position = copy.deepcopy(new_position)
        self.position = new_position
        self.camera_localization_received = True

    def ground_truth_position_callback(self, ground_truth):
        if self.use_camera_localization and self.camera_localization_received:
            return

        self.position = ground_truth.pose.pose.position

        orientation = ground_truth.pose.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        euler = tf_conversions.transformations.euler_from_quaternion(quaternion)
        self.theta = euler[2]

    def ur5_object_grasped_callback(self, message):
        if message.data:
            self.navigation_enabled = True
            rospy.loginfo("UR5 grasp signal received. BLUE navigation is now enabled.")

    def obstacles_callback(self, obstacles):
        pc_obstacles = pc2.read_points(
            obstacles,
            field_names=("x", "y", "z"),
            skip_nans=True
        )

        self.obstacles = []

        for point in pc_obstacles:
            x, y, z = point
            self.obstacles.append(geometry_msgs.msg.Point(x, y, z))

    def limits_callback(self, limits):
        pc_limits = pc2.read_points(
            limits,
            field_names=("x", "y", "z"),
            skip_nans=True
        )

        self.limits = []

        num_points = max(int(limits.width / 18), 0)
        count = 0

        for point in pc_limits:
            if count == 0:
                x, y, z = point
                self.limits.append(geometry_msgs.msg.Point(x, y, z))
                count = num_points
            else:
                count -= 1

    # -------------------------------------------------------------------------
    # Control action calculation.
    # -------------------------------------------------------------------------

    def controlActionCalculation(self):
        if self.position is None or self.goal_reached:
            return

        if not self.navigation_enabled:
            self.publish_stop_command()
            rospy.loginfo_throttle(3.0, "Waiting for UR5 grasp signal before moving.")
            return

        ackermann_control = ackermann_msgs.msg.AckermannDrive()
        ackermann_control.speed = 0.0
        ackermann_control.steering_angle = 0.0

        if self.distance(self.position, self.goal) < self.reached_distance:
            self.switch_to_next_target_or_finish()
            return

        goal_distance = self.distance(self.position, self.goal)

        if goal_distance - self.reached_distance < self.slow_down_distance:
            speed = MIN_SPEED
            self.local_target = self.global2local(self.goal)
        else:
            speed = MAX_SPEED

        nearest_front_obstacle_distance = self.nearest_front_obstacle_distance()

        if nearest_front_obstacle_distance is not None:
            if nearest_front_obstacle_distance < self.front_obstacle_slow_down_distance:
                speed = min(speed, self.obstacle_approach_speed)

        min_error = 1000.0
        best_command_found = False

        for steer in np.arange(-MAX_STEER_ANGLE, MAX_STEER_ANGLE + 0.001, self.delta_angle):
            if abs(steer) < 0.01:
                steer = 0.0

            k_sp = (MAX_STEER_ANGLE - abs(steer)) / MAX_STEER_ANGLE
            speed2 = max(speed * k_sp, self.creep_speed)

            if self.allow_reverse:
                directions = [-speed2, speed2]
            else:
                directions = [speed2]

            for candidate_speed in directions:
                flag_collision_risk = False
                minimum_trajectory_clearance = 1000.0

                if abs(steer) < 0.0001:
                    turn_radius = None
                    angular_velocity = 0.0
                else:
                    turn_radius = VEHICLE_LENGHT / tan(steer)
                    angular_velocity = candidate_speed / turn_radius

                final_local_point = geometry_msgs.msg.Point()
                final_orientation = 0.0

                for sample in np.arange(self.delta_sample, self.max_sample + 0.001, self.delta_sample):
                    local_point = geometry_msgs.msg.Point()

                    if abs(angular_velocity) < 0.0001:
                        local_point.x = candidate_speed * sample
                        local_point.y = 0.0
                        local_point.z = 0.0
                        final_orientation = 0.0
                    else:
                        local_point.x = candidate_speed * cos(angular_velocity * sample) * sample
                        local_point.y = candidate_speed * sin(angular_velocity * sample) * sample
                        local_point.z = 0.0
                        final_orientation = angular_velocity * sample

                    final_local_point = copy.deepcopy(local_point)

                    for obstacle in self.obstacles:
                        obstacle_distance_to_robot = sqrt(obstacle.x * obstacle.x + obstacle.y * obstacle.y)

                        if obstacle_distance_to_robot < self.robot_self_filter_radius:
                            continue

                        clearance = self.distance(local_point, obstacle)
                        minimum_trajectory_clearance = min(minimum_trajectory_clearance, clearance)

                        if clearance <= self.collision_margin:
                            flag_collision_risk = True
                            break

                    if flag_collision_risk:
                        break

                if not flag_collision_risk:
                    distance_error = self.distance(final_local_point, self.local_target)

                    desired_heading = self.angle(self.local_target, final_local_point)
                    orientation_error = abs(self.normalize_angle(desired_heading - final_orientation))

                    if minimum_trajectory_clearance >= 999.0:
                        clearance_error = 0.0
                    else:
                        clearance_error = 1.0 / max(minimum_trajectory_clearance, 0.10)

                    reverse_penalty = 0.0
                    if candidate_speed < 0.0:
                        reverse_penalty = self.reverse_motion_penalty

                    steering_penalty = self.steering_penalty_weight * abs(steer) / MAX_STEER_ANGLE

                    steering_change_penalty = (
                        self.steering_change_penalty_weight *
                        abs(steer - self.last_selected_steering) /
                        MAX_STEER_ANGLE
                    )

                    side_penalty = 0.0
                    if self.avoidance_side_sign != 0.0:
                        if steer * self.avoidance_side_sign < 0.0:
                            side_penalty = self.side_preference_weight

                    error = (
                        self.distance_error_weight * distance_error +
                        self.orientation_error_weight * orientation_error +
                        self.clearance_error_weight * clearance_error +
                        reverse_penalty +
                        steering_penalty +
                        steering_change_penalty +
                        side_penalty
                    )

                    if error < min_error:
                        min_error = error
                        best_command_found = True
                        ackermann_control.steering_angle = steer
                        ackermann_control.speed = candidate_speed

        if not best_command_found:
            if (
                self.avoidance_side_sign != 0.0 and
                (
                    nearest_front_obstacle_distance is None or
                    nearest_front_obstacle_distance > self.emergency_stop_distance
                )
            ):
                rospy.logwarn_throttle(
                    1.0,
                    "No sampled collision-free trajectory found. Using slow creep-turn recovery."
                )
                ackermann_control.speed = self.creep_speed
                ackermann_control.steering_angle = self.avoidance_side_sign * MAX_STEER_ANGLE
            else:
                rospy.logwarn_throttle(
                    1.0,
                    "No collision-free trajectory found. Stopping BLUE robot."
                )
                ackermann_control.speed = 0.0
                ackermann_control.steering_angle = 0.0

        self.last_selected_steering = ackermann_control.steering_angle
        self.last_selected_speed = ackermann_control.speed

        self.ackermann_command_publisher.publish(ackermann_control)

    # -------------------------------------------------------------------------
    # Local goal calculation.
    # -------------------------------------------------------------------------

    def localGoalCalculation(self):
        if self.position is None or self.goal_reached or self.limits is None:
            return

        if not self.navigation_enabled:
            return

        if len(self.limits) == 0:
            return

        goal_in_local_axis = self.global2local(self.goal)
        self.update_avoidance_side(goal_in_local_axis)

        wa, wr, ar, aa = 100.0, 1.0, 0.8, 0.3
        min_force = wr / pow(0.1, ar) - wa / pow(100.0, aa) + 1000.0

        local_goal = None
        self.local_path = []

        for limit_point in self.limits:
            if self.is_candidate_blocked_by_obstacle(limit_point):
                continue

            if self.avoidance_side_sign != 0.0:
                if limit_point.y * self.avoidance_side_sign < 0.20:
                    continue

            distance_a = self.distance(limit_point, goal_in_local_axis)

            if distance_a < 0.1:
                distance_a = 0.1

            force_a = wa / pow(distance_a, aa)

            min_distance = 10000.0
            force_r = 0.0

            for obstacle_point in self.obstacles:
                distance_r = self.distance(limit_point, obstacle_point)

                if distance_r < 0.1:
                    distance_r = 0.1

                if distance_r < min_distance:
                    min_distance = distance_r
                    force_r = wr / pow(distance_r, ar)

            force = force_r - force_a

            if self.avoidance_side_sign != 0.0:
                force -= 0.15 * abs(limit_point.y)

            if force < min_force:
                min_force = force
                local_goal = copy.deepcopy(limit_point)

        if local_goal is None:
            rospy.logwarn_throttle(
                1.0,
                "No valid local goal found in free-zone ring. Keeping previous local target."
            )
            return

        self.local_path.append(local_goal)

        wa2 = 3.0
        wr2 = 1.0
        ar2 = 0.5
        aa2 = 0.3
        radious = 4.0
        delta_rad = radious / 3.0

        selected_depth, selected_azimuth = self.cartesian2Spherical(local_goal.x, local_goal.y)

        for rad in np.arange(radious, delta_rad - 0.1, -delta_rad):
            ring_goal = None
            ring_min_force = wr2 / pow(0.1, ar2) - wa2 / pow(100.0, aa2) + 1000.0

            for limit_point in self.limits:
                depth, azimuth = self.cartesian2Spherical(limit_point.x, limit_point.y)
                p_in = self.spherical2Cartesian(rad, azimuth)

                if self.is_candidate_blocked_by_obstacle(p_in):
                    continue

                if self.avoidance_side_sign != 0.0:
                    if p_in.y * self.avoidance_side_sign < 0.05:
                        continue

                distance_a = self.distance(p_in, self.local_path[0])

                if distance_a < 0.1:
                    distance_a = 0.1

                force_a = wa2 / pow(distance_a, aa2)

                min_distance = 10000.0
                force_r = 0.0

                for obstacle_point in self.obstacles:
                    distance_r = self.distance(p_in, obstacle_point)

                    if distance_r < 0.1:
                        distance_r = 0.1

                    if distance_r < min_distance:
                        min_distance = distance_r
                        force_r = wr2 / pow(distance_r, ar2)

                force = force_r - force_a

                if self.avoidance_side_sign != 0.0:
                    force -= 0.10 * abs(p_in.y)

                if force < ring_min_force:
                    ring_min_force = force
                    ring_goal = copy.deepcopy(p_in)

            if ring_goal is None:
                ring_goal = self.spherical2Cartesian(rad, selected_azimuth)

            self.local_path.append(ring_goal)

        # Use the nearest inner local point as the immediate tracking target.
        # This makes the car start turning earlier instead of waiting until it is
        # almost in front of the obstacle.
        if len(self.local_path) >= 1:
            self.local_target = self.local_path[-1]
        else:
            self.local_target = self.global2local(self.goal)

        self.publish_local_path_markers()

    # -------------------------------------------------------------------------
    # Obstacle-side and candidate validation.
    # -------------------------------------------------------------------------

    def update_avoidance_side(self, goal_in_local_axis):
        nearest_front_obstacle = None
        nearest_front_distance = 10000.0

        for obstacle in self.obstacles:
            distance_to_robot = sqrt(obstacle.x * obstacle.x + obstacle.y * obstacle.y)

            if distance_to_robot < self.robot_self_filter_radius:
                continue

            if obstacle.x <= 0.30:
                continue

            if obstacle.x > self.front_obstacle_detection_distance:
                continue

            if abs(obstacle.y) > self.front_obstacle_sector_width:
                continue

            if distance_to_robot < nearest_front_distance:
                nearest_front_distance = distance_to_robot
                nearest_front_obstacle = obstacle

        if nearest_front_obstacle is None:
            if self.avoidance_side_sign != 0.0:
                rospy.loginfo("Front obstacle cleared. Resetting avoidance side.")
            self.avoidance_side_sign = 0.0
            return

        if self.avoidance_side_sign == 0.0:
            if abs(nearest_front_obstacle.y) < 0.20:
                if goal_in_local_axis.y >= 0.0:
                    self.avoidance_side_sign = 1.0
                else:
                    self.avoidance_side_sign = -1.0
            else:
                if nearest_front_obstacle.y > 0.0:
                    self.avoidance_side_sign = -1.0
                else:
                    self.avoidance_side_sign = 1.0

            rospy.loginfo(
                "Obstacle ahead detected. Selected avoidance side sign: %.1f",
                self.avoidance_side_sign
            )

    def is_candidate_blocked_by_obstacle(self, candidate_point):
        candidate_depth, candidate_angle = self.cartesian2Spherical(
            candidate_point.x,
            candidate_point.y
        )

        if candidate_depth < 0.001:
            return True

        for obstacle in self.obstacles:
            obstacle_depth, obstacle_angle = self.cartesian2Spherical(
                obstacle.x,
                obstacle.y
            )

            if obstacle_depth < self.robot_self_filter_radius:
                continue

            # Direct clearance check.
            # This prevents choosing local targets almost on top of the obstacle.
            if self.distance(candidate_point, obstacle) < self.candidate_clearance:
                return True

            # Ray-blocking check.
            # If an obstacle is between the robot and the candidate direction, the
            # candidate is considered unsafe.
            if obstacle_depth < candidate_depth - 0.20:
                angular_difference = abs(
                    self.normalize_angle(candidate_angle - obstacle_angle)
                )

                blocked_angle = atan2(self.blocked_sector_clearance, obstacle_depth)

                if angular_difference < blocked_angle:
                    return True

        return False

    def nearest_front_obstacle_distance(self):
        nearest_distance = None

        for obstacle in self.obstacles:
            obstacle_distance_to_robot = sqrt(obstacle.x * obstacle.x + obstacle.y * obstacle.y)

            if obstacle_distance_to_robot < self.robot_self_filter_radius:
                continue

            if obstacle.x <= 0.20:
                continue

            if abs(obstacle.y) > self.front_obstacle_sector_width:
                continue

            if nearest_distance is None or obstacle_distance_to_robot < nearest_distance:
                nearest_distance = obstacle_distance_to_robot

        return nearest_distance

    # -------------------------------------------------------------------------
    # Marker publication.
    # -------------------------------------------------------------------------

    def publish_local_path_markers(self):
        marker_msg = MarkerArray()

        for marker_id, point in enumerate(self.local_path):
            marker = Marker()
            marker.header.frame_id = "blue/velodyne"
            marker.header.stamp = rospy.Time()
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = point
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker_msg.markers.append(marker)

        self.marker_publisher.publish(marker_msg)

    # -------------------------------------------------------------------------
    # Goal management.
    # -------------------------------------------------------------------------

    def switch_to_next_target_or_finish(self):
        if self.current_target_index + 1 < len(self.target_points):
            self.current_target_index += 1
            self.goal = copy.deepcopy(self.target_points[self.current_target_index])
            self.local_target = self.global2local(self.goal)
            self.avoidance_side_sign = 0.0

            rospy.loginfo(
                "Intermediate goal reached. Switching to next goal x=%.3f, y=%.3f",
                self.goal.x,
                self.goal.y
            )

            return

        print("Goal reached")
        self.goal_reached = True
        self.publish_stop_command()
        self.blue_goal_reached_publisher.publish(std_msgs.msg.Bool(data=True))

    def publish_stop_command(self):
        ackermann_control = ackermann_msgs.msg.AckermannDrive()
        ackermann_control.speed = 0.0
        ackermann_control.steering_angle = 0.0
        self.ackermann_command_publisher.publish(ackermann_control)

    # -------------------------------------------------------------------------
    # Geometry utilities.
    # -------------------------------------------------------------------------

    def distance(self, p1, p2):
        return sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def angle(self, p1, p2):
        return atan2((p1.y - p2.y), p1.x - p2.x)

    def normalize_angle(self, angle):
        while angle > pi:
            angle -= 2.0 * pi

        while angle < -pi:
            angle += 2.0 * pi

        return angle

    def global2local(self, p):
        result = geometry_msgs.msg.Point()

        if self.position is None:
            result.x = p.x
            result.y = p.y
            result.z = 0.0
            return result

        x = p.x - self.position.x
        y = p.y - self.position.y

        result.x = x * cos(-self.theta) - y * sin(-self.theta)
        result.y = x * sin(-self.theta) + y * cos(-self.theta)
        result.z = 0.0

        return result

    def spherical2Cartesian(self, depth, azimuth):
        p = geometry_msgs.msg.Point()
        p.x = depth * cos(azimuth)
        p.y = depth * sin(azimuth)
        p.z = 0.0
        return p

    def cartesian2Spherical(self, x, y):
        depth = sqrt((x * x) + (y * y))
        azimuth = atan2(y, x)

        if azimuth < 0:
            azimuth += 2.0 * pi

        if azimuth >= 2.0 * pi:
            azimuth -= 2.0 * pi

        return depth, azimuth

    # -------------------------------------------------------------------------
    # Main loop.
    # -------------------------------------------------------------------------

    def run(self):
        rate = rospy.Rate(self.rate)

        count = 0

        while not rospy.is_shutdown():
            # Calculate the local path before the control command.
            # This avoids using an old frontal local target when an obstacle is
            # already visible in front of the robot.
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