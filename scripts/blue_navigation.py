#!/usr/bin/env python
#
# ROS node to calculate the trajectory towards the target while avoiding obstacles.

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
from math import pi, dist, cos, sin, fabs, sqrt, atan2, tan

#Vehicle specifications
MAX_STEER_ANGLE = 24.0 * pi / 180.0  # Radians
MAX_SPEED = 1.05                     # m/s. Fast enough for the practice, not excessive.
MIN_SPEED = 0.55                     # m/s. Avoids extremely slow motion.
VEHICLE_LENGHT = 1.05


class BlueTrajectoryPlanner(object):

    def __init__(self):
        super(BlueTrajectoryPlanner, self).__init__()

        ##ROS node initialization
        rospy.init_node("blue_planner_node", anonymous=True)

        #Parameter initialization
        self.delta_angle = rospy.get_param("~delta_angle_deg", 4.0) * pi / 180.0
        self.delta_sample = rospy.get_param("~trajectory_time_step", 0.15)
        self.max_sample = rospy.get_param("~trajectory_horizon", 1.25)
        self.reached_distance = rospy.get_param("~reached_distance", 0.70)
        self.slow_down_distance = rospy.get_param("~slow_down_distance", 1.60)
        self.rate = rospy.get_param("~control_rate", 10)

        #TODO Define other parameters needed for the planner.

        # -------------------------------------------------------------------------
        # BLOCK 1: Navigation and collision parameters.
        # The collision margin is intentionally moderate. This avoids exaggerated
        # detours and allows the vehicle to pass close to the obstacle, while still
        # keeping enough safety margin for the BLUE robot body.
        # -------------------------------------------------------------------------
        self.max_forward_speed = rospy.get_param("~max_forward_speed", MAX_SPEED)
        self.min_forward_speed = rospy.get_param("~min_forward_speed", MIN_SPEED)
        self.max_reverse_speed = rospy.get_param("~max_reverse_speed", 0.65)

        self.collision_margin = rospy.get_param("~collision_margin", 0.38)
        self.front_detection_distance = rospy.get_param("~front_detection_distance", 1.05)
        self.front_detection_width = rospy.get_param("~front_detection_width", 0.78)

        self.distance_error_weight = rospy.get_param("~distance_error_weight", 1.00)
        self.heading_error_weight = rospy.get_param("~heading_error_weight", 0.65)
        self.steering_penalty_weight = rospy.get_param("~steering_penalty_weight", 0.10)

        self.use_camera_localization = rospy.get_param("~use_camera_localization", False)
        self.use_ground_truth_fallback = rospy.get_param("~use_ground_truth_fallback", True)

        # -------------------------------------------------------------------------
        # BLOCK 2: Reactive close-obstacle recovery parameters.
        # When a near frontal obstacle is detected, the robot stops, reverses while
        # steering slightly, then moves forward again. If the obstacle is still in
        # front, the same sequence is repeated.
        # -------------------------------------------------------------------------
        self.recovery_stop_duration = rospy.get_param("~recovery_stop_duration", 0.35)
        self.recovery_reverse_duration = rospy.get_param("~recovery_reverse_duration", 1.10)
        self.recovery_forward_duration = rospy.get_param("~recovery_forward_duration", 0.85)

        self.recovery_reverse_speed = rospy.get_param("~recovery_reverse_speed", -0.62)
        self.recovery_forward_speed = rospy.get_param("~recovery_forward_speed", 0.78)
        self.recovery_steering_angle = rospy.get_param("~recovery_steering_angle", 0.30)

        #Variable initialization
        self.position = None
        self.theta = 0.0
        self.obstacles = []
        self.limits = None
        self.goal_reached = False #Wait until UR5 tells us to approach.

        #TODO Define other necessary variables.

        # -------------------------------------------------------------------------
        # BLOCK 3: Runtime state variables.
        # These variables store localization data, waypoint state, and the recovery
        # state machine used for close obstacle avoidance.
        # -------------------------------------------------------------------------
        self.last_camera_position = None
        self.last_camera_update_time = None
        self.camera_heading_min_displacement = rospy.get_param("~camera_heading_min_displacement", 0.05)

        self.recovery_state = "NORMAL"
        self.recovery_state_start_time = rospy.Time.now()
        self.recovery_steering_direction = 1.0
        self.recovery_cycle_counter = 0

        #TODO Target position initialization. It is possible to consider several target points to maneuver and approach the UR5 robot.

        # -------------------------------------------------------------------------
        # BLOCK 4: Target and optional test waypoint.
        # The final goal is the requested point. For the obstacle test, an optional
        # alignment waypoint makes the vehicle approach the obstacle from the left
        # side along y = -6.03, which makes the test repeatable and independent
        # from the red object and the UR5.
        # -------------------------------------------------------------------------
        final_goal = geometry_msgs.msg.Point()
        final_goal.x = rospy.get_param("~goal_x", 6.60)
        final_goal.y = rospy.get_param("~goal_y", -6.03)
        final_goal.z = 0.0

        self.target_points = []

        use_waypoint_approach = rospy.get_param("~use_waypoint_approach", False)

        if use_waypoint_approach:
            approach_goal = geometry_msgs.msg.Point()
            approach_goal.x = rospy.get_param("~approach_waypoint_x", 1.60)
            approach_goal.y = rospy.get_param("~approach_waypoint_y", -6.03)
            approach_goal.z = 0.0
            self.target_points.append(approach_goal)

        self.target_points.append(final_goal)

        self.current_target_index = 0
        self.goal = self.target_points[self.current_target_index]

        # Local path initialization (4 points) to avoid errors until the first point is calculated.
        self.local_path = [self.global_origin_point() for i in range(4)]

        print("Goal x: {}, y: {}".format(self.goal.x, self.goal.y))

        # TODO consider more subscribers/publishers if needed

        # -------------------------------------------------------------------------
        # BLOCK 5: Subscribers.
        # Ground truth is still available for testing. Camera localization can be
        # enabled with _use_camera_localization:=true, while ground truth remains a
        # fallback if desired.
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
                self.camera_pose_callback,
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

        rospy.loginfo("BLUE navigation planner started.")
        rospy.loginfo("Final goal: x=%.3f, y=%.3f", final_goal.x, final_goal.y)
        rospy.loginfo("Number of target points: %d", len(self.target_points))
        rospy.loginfo("Max forward speed: %.2f m/s", self.max_forward_speed)
        rospy.loginfo("Collision margin: %.2f m", self.collision_margin)

    #Callbacks
    #TODO modify this callback to not depend on the position given by Gazebo.
    def position_callback(self, ground_truth):
        # -------------------------------------------------------------------------
        # BLOCK 6: Ground-truth localization callback.
        # Used for reliable Part 2 testing. If camera localization is enabled and
        # fresh, ground truth is ignored unless it is needed as fallback.
        # -------------------------------------------------------------------------

        if self.use_camera_localization and self.last_camera_update_time is not None:
            camera_age = (rospy.Time.now() - self.last_camera_update_time).to_sec()
            if camera_age < 1.0:
                return

        self.position = ground_truth.pose.pose.position

        orientation = ground_truth.pose.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        euler = tf_conversions.transformations.euler_from_quaternion(quaternion)
        self.theta = euler[2]

    def camera_pose_callback(self, pose_array):
        # -------------------------------------------------------------------------
        # BLOCK 7: Approximate camera-based localization.
        # The external camera gives the BLUE position in /pose_array[1]. The heading
        # is approximated from the displacement between the current and previous
        # camera positions, which follows the localization requirement of the
        # practice without depending on Gazebo pose.
        # -------------------------------------------------------------------------

        if len(pose_array.poses) < 2:
            return

        detected_blue_position = pose_array.poses[1].position

        if detected_blue_position.z < 0.0:
            return

        new_position = geometry_msgs.msg.Point()
        new_position.x = detected_blue_position.x
        new_position.y = detected_blue_position.y
        new_position.z = 0.0

        if self.last_camera_position is not None:
            dx = new_position.x - self.last_camera_position.x
            dy = new_position.y - self.last_camera_position.y
            displacement = sqrt(dx * dx + dy * dy)

            if displacement >= self.camera_heading_min_displacement:
                self.theta = atan2(dy, dx)

        self.position = new_position
        self.last_camera_position = copy.deepcopy(new_position)
        self.last_camera_update_time = rospy.Time.now()

    def obstacles_callback(self, obstacles):
        # -------------------------------------------------------------------------
        # BLOCK 8: Obstacle point callback.
        # The planner stores obstacle points in the local Velodyne frame. These are
        # used both by the trajectory collision checker and by the close-obstacle
        # recovery state machine.
        # -------------------------------------------------------------------------

        pc_obstacles = pc2.read_points(obstacles, field_names=("x", "y", "z"), skip_nans=True)

        #Save as geometry_msg.Point
        self.obstacles = []

        for point in pc_obstacles:
            x, y, z = point
            new_point = geometry_msgs.msg.Point()
            new_point.x = x
            new_point.y = y
            new_point.z = z
            self.obstacles.append(new_point)

    def limits_callback(self, limits):
        # -------------------------------------------------------------------------
        # BLOCK 9: Free-zone callback.
        # The free-zone point cloud is downsampled to keep local target computation
        # fast enough for online navigation.
        # -------------------------------------------------------------------------

        pc_limits = pc2.read_points(limits, field_names=("x", "y", "z"), skip_nans=True)

        #Save as geometry_msg.Point
        self.limits = []

        #Decrease the size of the point cloud to speed up the search.
        num_points = max(int(limits.width / 18), 0)
        count = 0

        for point in pc_limits:
            if count == 0:
                x, y, z = point
                new_point = geometry_msgs.msg.Point()
                new_point.x = x
                new_point.y = y
                new_point.z = z
                self.limits.append(new_point)
                count = num_points
            else:
                count -= 1

    # Trajectory planning towards the target while avoiding obstacles. It is executed at a lower frequency.
    ''' Code implemented from the following research article:
            OpenStreetMap-Based Autonomous Navigation With LiDAR Naive-Valley-Path Obstacle Avoidance
            Miguel Ángel Muñoz Bañón, Edison Velasco Sánchez, Francisco A. Candelas, Fernando Torres
            IEEE Transactions on Intelligent Transportation Systems, 2022
    '''
    def localGoalCalculation(self):
        if self.position is None or self.goal_reached:
            return

        # -------------------------------------------------------------------------
        # BLOCK 10: Fallback local path.
        # If /free_zone has not arrived yet, the planner uses the global goal
        # transformed into the local frame.
        # -------------------------------------------------------------------------

        if self.limits is None or len(self.limits) == 0:
            local_goal = self.global2local(self.goal)
            self.local_path = [local_goal for i in range(4)]
            self.publish_local_path_markers()
            return

        #Parameter definition
        wa, wr, ar, aa = 100.0, 1.0, 0.8, 0.3
        min_force = wr / pow(0.1, ar) - wa / pow(100.0, aa) + 1000.0

        local_goal = geometry_msgs.msg.Point()
        self.local_path = []
        goal_in_local_axis = self.global2local(self.goal)

        # 1) Trajectory points within the limited radius.
        for limit_point in self.limits:
            distance_a = self.distance(limit_point, goal_in_local_axis)
            if distance_a < 0.1:
                distance_a = 0.1

            force_a = wa / pow(distance_a, aa)

            #Computation of the minimum obstacle distance and its weight.
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

            if force < min_force:
                min_force = force
                local_goal = limit_point

        self.local_path.append(local_goal)

        # 2) Trajectory points with the inner rings.
        # Parameter definition.
        wa2 = 3.0
        wr2 = 1.0
        ar2 = 0.5
        aa2 = 0.3
        radious = 4.0
        delta_rad = radious / 3.0

        for rad in np.arange(radious, delta_rad - 0.1, -delta_rad):
            min_force = wr / pow(0.1, ar) - wa / pow(100.0, aa) + 1000.0
            local_goal = self.global2local(self.goal)

            for limit_point in self.limits:
                depth, azimuth = self.cartesian2Spherical(limit_point.x, limit_point.y)
                p_in = self.spherical2Cartesian(rad, azimuth)

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

                if force < min_force:
                    min_force = force
                    local_goal = p_in

            self.local_path.append(local_goal)

        #Publish the target visualization.
        self.publish_local_path_markers()

    def publish_local_path_markers(self):
        # -------------------------------------------------------------------------
        # BLOCK 11: Local path visualization.
        # Publishes purple spheres in RViz to show the selected local path points.
        # -------------------------------------------------------------------------

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

    # Calculate the control commands to reach the planned local target.
    def controlActionCalculation(self):
        # Detect if the trajectory is finished.
        if self.position is None or self.goal_reached:
            self.publish_ackermann_command(0.0, 0.0)
            return

        # -------------------------------------------------------------------------
        # BLOCK 12: Close-obstacle reactive layer.
        # This layer starts only when an obstacle is near the frontal safety zone.
        # It prevents the exaggerated far-away detours produced by relying only on
        # a large local planner radius.
        # -------------------------------------------------------------------------

        near_front_obstacle, nearest_front_distance = self.detect_near_front_obstacle()

        if self.recovery_state == "NORMAL" and near_front_obstacle:
            rospy.logwarn(
                "Near frontal obstacle detected at %.2f m. Starting stop-reverse-forward recovery.",
                nearest_front_distance
            )
            self.start_recovery_state("STOP_BEFORE_REVERSING")
            self.publish_ackermann_command(0.0, 0.0)
            return

        if self.recovery_state != "NORMAL":
            self.execute_recovery_state_machine()
            return

        # Detect if the robot has reached the following target point.
        if self.distance(self.position, self.goal) < self.reached_distance:
            if self.current_target_index + 1 < len(self.target_points):
                self.current_target_index += 1
                self.goal = self.target_points[self.current_target_index]

                rospy.loginfo(
                    "Intermediate target reached. Switching to target %d: x=%.3f, y=%.3f",
                    self.current_target_index,
                    self.goal.x,
                    self.goal.y
                )

                self.localGoalCalculation()
                self.publish_ackermann_command(0.0, 0.0)
                return

            print("Goal reached")
            self.goal_reached = True
            self.publish_ackermann_command(0.0, 0.0)
            return

        # -------------------------------------------------------------------------
        # BLOCK 13: Speed selection.
        # The robot slows down near the goal but is prevented from becoming
        # excessively slow during normal navigation.
        # -------------------------------------------------------------------------

        goal_distance = self.distance(self.position, self.goal)

        if goal_distance - self.reached_distance < self.slow_down_distance:
            speed = self.min_forward_speed
            #Set the target as local target.
            if len(self.local_path) >= 2:
                self.local_path[-2] = self.global2local(self.goal)
        else:
            speed = self.max_forward_speed

        #Variable initialization
        min_error = 100000.0
        best_command_found = False

        local_target = self.get_active_local_target()

        ackermann_control = ackermann_msgs.msg.AckermannDrive()
        ackermann_control.speed = 0.0
        ackermann_control.steering_angle = 0.0

        # -------------------------------------------------------------------------
        # BLOCK 14: Ackermann trajectory sampling and evaluation.
        # Candidate steering commands are simulated using the bicycle model. Any
        # trajectory whose sampled points pass too close to obstacle points is
        # discarded. The remaining trajectory with the best distance + orientation
        # score is selected.
        # -------------------------------------------------------------------------

        # Check possible action control commands
        for steer in np.arange(-MAX_STEER_ANGLE, MAX_STEER_ANGLE + 0.01, self.delta_angle):
            if abs(steer) < 0.01:
                steer = 0.0

            #Reduce the velocity when the turn is big.
            k_sp = (MAX_STEER_ANGLE - abs(steer)) / MAX_STEER_ANGLE
            speed2 = max(speed * k_sp, self.min_forward_speed)

            # Normal navigation is forward-only. Reverse is reserved for the
            # explicit close-obstacle recovery state machine.
            directions = [speed2]

            for dir in directions:
                flag_collision_risk = False
                sampled_trajectory = []
                final_heading = 0.0

                #TODO Calculate turn radius and velocity using the robot kinematics.

                # -------------------------------------------------------------------------
                # BLOCK 15: Bicycle kinematics.
                # For each sampled time, predict the local pose obtained by applying
                # speed dir and steering angle steer.
                # -------------------------------------------------------------------------

                # Calculate the trajectory using the angle rotation. Several points are sampled from the trajectory over time,
                # therefore the sample variable is equivalent to the t variable in the robot kinematic equation.
                for sample in np.arange(self.delta_sample, self.max_sample + 0.01, self.delta_sample):
                    local_point = geometry_msgs.msg.Point()

                    #TODO calculate trajectory points using the robot kinematics.
                    local_point, final_heading = self.predict_bicycle_model_point(dir, steer, sample)
                    sampled_trajectory.append(local_point)

                    #TODO Detect collisions risk.
                    for obstacle in self.obstacles:
                        if self.distance(local_point, obstacle) < self.collision_margin:
                            flag_collision_risk = True
                            break

                    if flag_collision_risk:
                        break

                # If there is no collision, evaluate the trajectory.
                if not flag_collision_risk and len(sampled_trajectory) > 0:
                    #TODO estimate the trajectory evaluation in terms of the distance and orientation error to the local target (self.local_path[-2]).

                    # -------------------------------------------------------------------------
                    # BLOCK 16: Trajectory scoring.
                    # The final point is scored by distance to the local target, heading
                    # alignment, and a small steering penalty to avoid unnecessary wide turns.
                    # -------------------------------------------------------------------------

                    final_point = sampled_trajectory[-1]
                    error = self.evaluate_trajectory(final_point, final_heading, local_target, steer)

                    if error < min_error:
                        min_error = error
                        ackermann_control.steering_angle = steer
                        ackermann_control.speed = dir
                        best_command_found = True

        # -------------------------------------------------------------------------
        # BLOCK 17: Command publication or safe fallback.
        # If no trajectory is safe, the robot enters the same stop-reverse-forward
        # recovery behavior used for close frontal obstacles.
        # -------------------------------------------------------------------------

        if not best_command_found:
            rospy.logwarn_throttle(1.0, "No safe forward trajectory found. Starting recovery.")
            self.start_recovery_state("STOP_BEFORE_REVERSING")
            self.publish_ackermann_command(0.0, 0.0)
            return

        #Publish message
        self.ackermann_command_publisher.publish(ackermann_control)

    def predict_bicycle_model_point(self, velocity, steering_angle, time_value):
        # -------------------------------------------------------------------------
        # BLOCK 18: Bicycle model trajectory point.
        # The robot starts at local pose (0, 0, 0). Positive x is forward and
        # positive y is left in the local Velodyne/base frame.
        # -------------------------------------------------------------------------

        local_point = geometry_msgs.msg.Point()

        if abs(steering_angle) < 1e-4:
            local_point.x = velocity * time_value
            local_point.y = 0.0
            local_point.z = 0.0
            final_heading = 0.0
            return local_point, final_heading

        turn_radius = VEHICLE_LENGHT / tan(steering_angle)
        angular_velocity = velocity / turn_radius

        final_heading = angular_velocity * time_value

        local_point.x = turn_radius * sin(final_heading)
        local_point.y = turn_radius * (1.0 - cos(final_heading))
        local_point.z = 0.0

        return local_point, final_heading

    def evaluate_trajectory(self, final_point, final_heading, local_target, steering_angle):
        # -------------------------------------------------------------------------
        # BLOCK 19: Trajectory objective function.
        # A good trajectory ends near the local target and points toward it. The
        # steering penalty discourages unnecessarily wide obstacle avoidance.
        # -------------------------------------------------------------------------

        distance_error = self.distance(final_point, local_target)

        target_angle = atan2(
            local_target.y - final_point.y,
            local_target.x - final_point.x
        )

        heading_error = abs(self.normalize_angle(target_angle - final_heading))
        steering_penalty = abs(steering_angle)

        return (
            self.distance_error_weight * distance_error +
            self.heading_error_weight * heading_error +
            self.steering_penalty_weight * steering_penalty
        )

    def get_active_local_target(self):
        # -------------------------------------------------------------------------
        # BLOCK 20: Active local target selection.
        # The second-to-last local path point is used as in the original template.
        # If it is not available, the global goal transformed to the robot frame is
        # used.
        # -------------------------------------------------------------------------

        if self.local_path is not None and len(self.local_path) >= 2:
            return self.local_path[-2]

        return self.global2local(self.goal)

    def detect_near_front_obstacle(self):
        # -------------------------------------------------------------------------
        # BLOCK 21: Near frontal obstacle detector.
        # It only reacts when the obstacle is really close and in front of the car.
        # This is the key to avoiding exaggerated early detours.
        # -------------------------------------------------------------------------

        nearest_distance = 10000.0
        detected = False

        for obstacle in self.obstacles:
            if obstacle.x < 0.20:
                continue

            if obstacle.x > self.front_detection_distance:
                continue

            if abs(obstacle.y) > self.front_detection_width:
                continue

            obstacle_distance = sqrt(obstacle.x * obstacle.x + obstacle.y * obstacle.y)

            if obstacle_distance < nearest_distance:
                nearest_distance = obstacle_distance

            detected = True

        return detected, nearest_distance

    def start_recovery_state(self, new_state):
        # -------------------------------------------------------------------------
        # BLOCK 22: Recovery state start.
        # The steering side is selected from the obstacle distribution. If the
        # obstacle is centered, the side alternates between recovery cycles.
        # -------------------------------------------------------------------------

        self.recovery_state = new_state
        self.recovery_state_start_time = rospy.Time.now()
        self.recovery_cycle_counter += 1
        self.recovery_steering_direction = self.choose_recovery_steering_direction()

    def choose_recovery_steering_direction(self):
        # -------------------------------------------------------------------------
        # BLOCK 23: Recovery steering selection.
        # If the obstacle is more visible on the left, the robot steers right while
        # reversing. If it is more visible on the right, it steers left. If centered,
        # it alternates sides to avoid getting stuck.
        # -------------------------------------------------------------------------

        frontal_y_values = []

        for obstacle in self.obstacles:
            if 0.20 <= obstacle.x <= self.front_detection_distance and abs(obstacle.y) <= self.front_detection_width:
                frontal_y_values.append(obstacle.y)

        if len(frontal_y_values) == 0:
            return 1.0 if self.recovery_cycle_counter % 2 == 0 else -1.0

        mean_y = sum(frontal_y_values) / float(len(frontal_y_values))

        if abs(mean_y) < 0.08:
            return 1.0 if self.recovery_cycle_counter % 2 == 0 else -1.0

        if mean_y > 0.0:
            return -1.0

        return 1.0

    def execute_recovery_state_machine(self):
        # -------------------------------------------------------------------------
        # BLOCK 24: Stop-reverse-forward obstacle avoidance.
        # Required behavior:
        #   1) Stop when the obstacle is detected.
        #   2) Reverse while steering slightly.
        #   3) Move forward again.
        #   4) Repeat if the obstacle is detected again.
        # -------------------------------------------------------------------------

        elapsed = (rospy.Time.now() - self.recovery_state_start_time).to_sec()

        if self.recovery_state == "STOP_BEFORE_REVERSING":
            self.publish_ackermann_command(0.0, 0.0)

            if elapsed >= self.recovery_stop_duration:
                self.recovery_state = "REVERSING_WITH_STEERING"
                self.recovery_state_start_time = rospy.Time.now()

            return

        if self.recovery_state == "REVERSING_WITH_STEERING":
            reverse_steering = self.recovery_steering_direction * self.recovery_steering_angle
            self.publish_ackermann_command(self.recovery_reverse_speed, reverse_steering)

            if elapsed >= self.recovery_reverse_duration:
                self.recovery_state = "MOVING_FORWARD_AFTER_REVERSE"
                self.recovery_state_start_time = rospy.Time.now()

            return

        if self.recovery_state == "MOVING_FORWARD_AFTER_REVERSE":
            near_front_obstacle, nearest_front_distance = self.detect_near_front_obstacle()

            if elapsed > 0.45 and near_front_obstacle:
                rospy.logwarn(
                    "Obstacle still detected after recovery at %.2f m. Repeating recovery.",
                    nearest_front_distance
                )
                self.start_recovery_state("STOP_BEFORE_REVERSING")
                self.publish_ackermann_command(0.0, 0.0)
                return

            forward_steering = -0.55 * self.recovery_steering_direction * self.recovery_steering_angle
            self.publish_ackermann_command(self.recovery_forward_speed, forward_steering)

            if elapsed >= self.recovery_forward_duration:
                near_front_obstacle, nearest_front_distance = self.detect_near_front_obstacle()

                if near_front_obstacle:
                    rospy.logwarn(
                        "Obstacle detected again at %.2f m. Repeating recovery.",
                        nearest_front_distance
                    )
                    self.start_recovery_state("STOP_BEFORE_REVERSING")
                    self.publish_ackermann_command(0.0, 0.0)
                else:
                    rospy.loginfo("Recovery completed. Returning to normal navigation.")
                    self.recovery_state = "NORMAL"
                    self.recovery_state_start_time = rospy.Time.now()

            return

        self.recovery_state = "NORMAL"

    def publish_ackermann_command(self, speed, steering_angle):
        # -------------------------------------------------------------------------
        # BLOCK 25: Safe Ackermann command publication.
        # Speeds and steering angles are clamped before publication.
        # -------------------------------------------------------------------------

        command = ackermann_msgs.msg.AckermannDrive()

        command.speed = max(
            -self.max_reverse_speed,
            min(self.max_forward_speed, speed)
        )

        command.steering_angle = max(
            -MAX_STEER_ANGLE,
            min(MAX_STEER_ANGLE, steering_angle)
        )

        self.ackermann_command_publisher.publish(command)

    def distance(self, p1, p2):
        return sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    # Angle from point p2 to p1
    def angle(self, p1, p2):
        return atan2((p1.y - p2.y), p1.x - p2.x)

    #Transformation from global to local position
    def global2local(self, p):
        result = geometry_msgs.msg.Point()

        if self.position is None:
            return result

        #Translation
        x = (p.x - self.position.x)
        y = (p.y - self.position.y)

        #Rotation
        result.x = x * cos(-self.theta) - y * sin(-self.theta)
        result.y = x * sin(-self.theta) + y * cos(-self.theta)
        result.z = 0.0

        return result

    #From spherical to cartesian coordinates
    def spherical2Cartesian(self, depth, azimuth):
        sin_azimuth = sin(azimuth)
        cos_azimuth = cos(azimuth)
        p = geometry_msgs.msg.Point()
        p.x = depth * cos_azimuth
        p.y = depth * sin_azimuth
        p.z = 0.0
        return p

    #From cartesian to spherical coordinates
    def cartesian2Spherical(self, x, y):
        depth = sqrt((x * x) + (y * y))
        azimuth = atan2(y, x)

        if azimuth < 0:
            azimuth += 2 * pi

        if azimuth >= 2 * pi:
            azimuth -= 2 * pi

        return depth, azimuth

    def normalize_angle(self, angle_value):
        # -------------------------------------------------------------------------
        # BLOCK 26: Angle normalization.
        # Keeps angular errors inside [-pi, pi].
        # -------------------------------------------------------------------------

        while angle_value > pi:
            angle_value -= 2.0 * pi

        while angle_value < -pi:
            angle_value += 2.0 * pi

        return angle_value

    def global_origin_point(self):
        point = geometry_msgs.msg.Point()
        point.x = 0.0
        point.y = 0.0
        point.z = 0.0
        return point

    def run(self):
        #Control loop
        rate = rospy.Rate(self.rate)
        count = 3

        while not rospy.is_shutdown():
            self.controlActionCalculation()

            if count == 0:
                self.localGoalCalculation()
                count = 3
            else:
                count -= 1

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
