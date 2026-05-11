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
MAX_STEER_ANGLE = 24.0*pi/180.0 # Radians
MAX_SPEED = 1.10    # m/s. Moderate speed: fast enough for the test, not aggressive.
MIN_SPEED = 0.48
VEHICLE_LENGHT = 1.05

class BlueTrajectoryPlanner(object):

    def __init__(self):
        super(BlueTrajectoryPlanner, self).__init__()

        ##ROS node initialization
        rospy.init_node("blue_planner_node", anonymous=True)

        #Parameter initialization
        self.delta_angle = rospy.get_param("~steering_sample_step_deg", 6.0) * pi / 180.0
        self.delta_sample = rospy.get_param("~trajectory_time_step", 0.20)
        self.max_sample = rospy.get_param("~trajectory_horizon", 1.10)
        self.reached_distance = rospy.get_param("~reached_distance", 0.70)
        self.slow_down_distance = rospy.get_param("~slow_down_distance", 1.80)
        self.rate = rospy.get_param("~control_rate", 10.0)

        #TODO Define other parameters needed for the planner.
        # -------------------------------------------------------------------------
        # BLOCK 1: Planner and obstacle-avoidance parameters.
        # These values keep the robot close to the obstacle without starting the
        # avoidance too early. The obstacle recovery behaviour is intentionally local:
        # stop, reverse while steering, then drive forward again.
        # -------------------------------------------------------------------------
        self.max_navigation_speed = rospy.get_param("~max_navigation_speed", MAX_SPEED)
        self.min_navigation_speed = rospy.get_param("~min_navigation_speed", MIN_SPEED)
        self.collision_margin = rospy.get_param("~collision_margin", 0.38)
        self.local_planner_radius = rospy.get_param("~local_planner_radius", 2.80)

        self.front_obstacle_min_distance = rospy.get_param("~front_obstacle_min_distance", 0.35)
        self.front_obstacle_stop_distance = rospy.get_param("~front_obstacle_stop_distance", 1.20)
        self.front_obstacle_half_width = rospy.get_param("~front_obstacle_half_width", 0.48)

        self.emergency_stop_duration = rospy.get_param("~emergency_stop_duration", 0.35)
        self.reverse_recovery_duration = rospy.get_param("~reverse_recovery_duration", 1.10)
        self.forward_recovery_duration = rospy.get_param("~forward_recovery_duration", 1.20)
        self.recovery_reverse_speed = rospy.get_param("~recovery_reverse_speed", 0.55)
        self.recovery_forward_speed = rospy.get_param("~recovery_forward_speed", 0.78)
        self.recovery_steering_angle = rospy.get_param("~recovery_steering_angle", 0.32)

        self.use_camera_localization = rospy.get_param("~use_camera_localization", False)
        self.use_ground_truth_fallback = rospy.get_param("~use_ground_truth_fallback", True)
        self.camera_localization_timeout = rospy.get_param("~camera_localization_timeout", 1.00)

        #Variable initialization
        self.position = None
        self.theta = 0
        self.obstacles = []
        self.limits = None
        self.goal_reached = False #Wait until UR5 tells us to approach.
        #TODO Define other necessary variables.
        # -------------------------------------------------------------------------
        # BLOCK 2: Runtime state variables.
        # recovery_state controls the requested behaviour when a close frontal
        # obstacle is detected: STOP -> REVERSE_TURN -> FORWARD_RECOVERY -> NORMAL.
        # -------------------------------------------------------------------------
        self.recovery_state = "NORMAL"
        self.recovery_state_start_time = rospy.Time.now()
        self.recovery_turn_sign = -1.0
        self.last_camera_position = None
        self.last_camera_time = None
        self.active_goal_index = 0

        #TODO Target position initialization. It is possible to consider several target points to maneuver and approach the UR5 robot.
        # -------------------------------------------------------------------------
        # BLOCK 3: Target-point initialization.
        # The final target is configurable from rosrun. For the requested obstacle
        # test, use_test_waypoints adds a pre-obstacle waypoint on the same Y line,
        # so the obstacle at (3.10, -6.03) is actually encountered before the final
        # goal at (6.60, -6.03).
        # -------------------------------------------------------------------------
        final_goal = geometry_msgs.msg.Point()
        final_goal.x = rospy.get_param("~goal_x", 6.60)
        final_goal.y = rospy.get_param("~goal_y", -6.03)
        final_goal.z = 0.0

        self.use_test_waypoints = rospy.get_param("~use_test_waypoints", False)
        self.waypoints = []

        if self.use_test_waypoints:
            pre_obstacle_waypoint = geometry_msgs.msg.Point()
            pre_obstacle_waypoint.x = rospy.get_param("~pre_obstacle_waypoint_x", 2.05)
            pre_obstacle_waypoint.y = rospy.get_param("~pre_obstacle_waypoint_y", -6.03)
            pre_obstacle_waypoint.z = 0.0
            self.waypoints.append(pre_obstacle_waypoint)

        self.waypoints.append(final_goal)
        self.goal = copy.deepcopy(self.waypoints[self.active_goal_index])

        # Local path initialization (4 points) to avoid errors until the first point is calculated.
        self.local_path=[self.goal for i in range(4)]
        print("Goal x: {}, y: {}".format(self.goal.x, self.goal.y))

        # TODO consider more subscribers/publishers if needed
        # Subscribers definition
        if self.use_ground_truth_fallback:
            self.position_subscriber = rospy.Subscriber("/blue/ground_truth",
                Odometry, self.position_callback, queue_size=1)

        if self.use_camera_localization:
            self.camera_position_subscriber = rospy.Subscriber("/pose_array",
                geometry_msgs.msg.PoseArray, self.camera_pose_callback, queue_size=1)

        self.obstacles_subscriber = rospy.Subscriber("/obstacles",
            sensor_msgs.msg.PointCloud2, self.obstacles_callback, queue_size=1)
        self.free_zone_subscriber = rospy.Subscriber("/free_zone",
            sensor_msgs.msg.PointCloud2, self.limits_callback, queue_size=1)

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
            "/blue/navigation_finished",
            std_msgs.msg.Bool,
            queue_size=10,
        )

        rospy.on_shutdown(self.publish_stop_command)

        rospy.loginfo("BLUE trajectory planner started.")
        rospy.loginfo("Final goal: x=%.3f, y=%.3f", final_goal.x, final_goal.y)
        rospy.loginfo("Use test waypoints: %s", str(self.use_test_waypoints))
        rospy.loginfo("Speed range: min=%.2f m/s, max=%.2f m/s", self.min_navigation_speed, self.max_navigation_speed)
        rospy.loginfo("Collision margin: %.2f m", self.collision_margin)

    #Callbacks
    #TODO modify this callback to not depend on the position given by Gazebo.
    def position_callback(self, ground_truth):
        # -------------------------------------------------------------------------
        # BLOCK 4: Ground-truth localization fallback.
        # The practice asks to approximate localization with the camera. For robust
        # testing, this callback remains available as a fallback. If camera
        # localization is active and recent, the Gazebo pose is ignored.
        # -------------------------------------------------------------------------
        if self.use_camera_localization and self.last_camera_time is not None:
            camera_age = (rospy.Time.now() - self.last_camera_time).to_sec()
            if camera_age <= self.camera_localization_timeout:
                return

        self.position = ground_truth.pose.pose.position
        quaternion = [
            ground_truth.pose.pose.orientation.x,
            ground_truth.pose.pose.orientation.y,
            ground_truth.pose.pose.orientation.z,
            ground_truth.pose.pose.orientation.w,
        ]
        euler = tf_conversions.transformations.euler_from_quaternion(quaternion)
        self.theta = euler[2]

    def camera_pose_callback(self, pose_array):
        # -------------------------------------------------------------------------
        # BLOCK 5: Camera-based BLUE localization approximation.
        # object_localization.py publishes the red object in poses[0] and BLUE in
        # poses[1]. The x/y camera coordinates are used as global top-view
        # coordinates. The heading is approximated from consecutive positions.
        # -------------------------------------------------------------------------
        if len(pose_array.poses) < 2:
            return

        detected_blue_position = pose_array.poses[1].position
        if detected_blue_position.z < 0:
            return

        current_position = geometry_msgs.msg.Point()
        current_position.x = detected_blue_position.x
        current_position.y = detected_blue_position.y
        current_position.z = 0.0

        current_time = rospy.Time.now()

        if self.last_camera_position is not None:
            dx = current_position.x - self.last_camera_position.x
            dy = current_position.y - self.last_camera_position.y
            displacement = sqrt(dx * dx + dy * dy)
            if displacement > 0.05:
                self.theta = atan2(dy, dx)

        self.position = current_position
        self.last_camera_position = copy.deepcopy(current_position)
        self.last_camera_time = current_time

    def obstacles_callback(self, obstacles):
        pc_obstacles=pc2.read_points(obstacles, field_names=("x", "y", "z"), skip_nans=True)
        #Save as geometry_msg.Point
        self.obstacles=[]
        for point in pc_obstacles:
            x,y,z = point
            new_point = geometry_msgs.msg.Point(x,y,z)
            self.obstacles.append(new_point)

    def limits_callback(self, limits):
        pc_limits=pc2.read_points(limits, field_names=("x", "y", "z"), skip_nans=True)
        #Save as geometry_msg.Point
        self.limits=[]

        #Decrease the size of the point cloud to speed up the search.
        num_points = max(int(limits.width/18),0)
        count=0
        for point in pc_limits:
            if count==0:
                x,y,z = point
                new_point = geometry_msgs.msg.Point(x,y,z)
                self.limits.append(new_point)
                count=num_points
            else: count-=1

    # Trajectory planning towards the target while avoiding obstacles. It is executed at a lower frequency.
    ''' Code implemented from the following research article:
            OpenStreetMap-Based Autonomous Navigation With LiDAR Naive-Valley-Path Obstacle Avoidance
            Miguel Ángel Muñoz Bañón, Edison Velasco Sánchez, Francisco A. Candelas, Fernando Torres
            IEEE Transactions on Intelligent Transportation Systems, 2022
    '''
    def localGoalCalculation(self):
        if self.position is None or self.goal_reached or self.limits is None:
            return
        if len(self.limits) == 0:
            return

        #Parameter definition
        wa, wr, ar, aa = 100.0, 1.0, 0.8, 0.3
        min_force = wr / pow(0.1, ar) - wa / pow(100.0, aa) + 1000

        local_goal=geometry_msgs.msg.Point()
        self.local_path=[]
        goal_in_local_axis = self.global2local(self.goal)
        # 1) Trajectory points within the limited radius.
        for limit_point in self.limits:
            distance_a = self.distance(limit_point, goal_in_local_axis)
            if distance_a<0.1:
                distance_a=0.1

            force_a = wa / pow(distance_a,aa)

            #Computation of the minimum obstacle distance and its weight.
            min_distance=10000.0
            force_r = 0.0
            for obstacle_point in self.obstacles:
                distance_r = self.distance(limit_point,obstacle_point)
                if distance_r<0.1:
                    distance_r=0.1
                if distance_r<min_distance:
                    min_distance=distance_r
                    force_r = wr / pow(distance_r, ar)

            force = force_r - force_a

            if force < min_force:
                min_force=force
                local_goal=limit_point
        self.local_path.append(local_goal)
        # 2) Trajectory points with the inner rings.
        # Parameter definition.
        wa2 = 3.0
        wr2 = 1.0
        ar2 = 0.5
        aa2 = 0.3
        radious = self.local_planner_radius
        delta_rad = radious / 3.0

        for rad in np.arange(radious, delta_rad-0.1, -delta_rad):
            min_force = wr / pow(0.1, ar) - wa / pow(100.0, aa) + 1000
            #Computation of the minimum obstacle distance and its weight.
            for limit_point in self.limits:
                depth, azimuth=self.cartesian2Spherical(limit_point.x, limit_point.y)
                p_in  = self.spherical2Cartesian(rad, azimuth)

                distance_a=self.distance(p_in, self.local_path[0])
                if distance_a<0.1:
                    distance_a=0.1
                force_a = wa2 /pow(distance_a,aa2)

                min_distance = 10000.0
                force_r = 0.0
                for obstacle_point in self.obstacles:
                    distance_r = self.distance(p_in, obstacle_point)
                    if distance_r<0.1:
                        distance_r=0.1
                    if distance_r<min_distance:
                        min_distance=distance_r
                        force_r = wr2 / pow(distance_r, ar2)

                force = force_r-force_a
                if force < min_force:
                    min_force=force
                    local_goal=p_in
            self.local_path.append(local_goal)

        #Publish the target visualization.
        marker_msg = MarkerArray()
        for id, point in enumerate(self.local_path):
            marker = Marker()
            marker.header.frame_id = "blue/velodyne"
            marker.header.stamp = rospy.Time()
            marker.id = id
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

        if self.recovery_state != "NORMAL":
            self.execute_recovery_state_machine()
            return

        obstacle_found, nearest_obstacle = self.detect_close_frontal_obstacle()
        if obstacle_found:
            self.start_recovery_state_machine(nearest_obstacle)
            self.publish_ackermann_command(0.0, 0.0)
            return

        ackermann_control=ackermann_msgs.msg.AckermannDrive()
        ackermann_control.speed, ackermann_control.steering_angle = 0.0, 0.0
        # Detect if the robot has reached the following target point.
        if self.distance(self.position, self.goal) < self.reached_distance:
            if self.active_goal_index < len(self.waypoints) - 1:
                self.active_goal_index += 1
                self.goal = copy.deepcopy(self.waypoints[self.active_goal_index])
                rospy.loginfo("Intermediate waypoint reached. New goal x=%.3f, y=%.3f", self.goal.x, self.goal.y)
                return

            print("Goal reached")
            self.goal_reached=True
            self.ackermann_command_publisher.publish(ackermann_control)
            self.navigation_finished_publisher.publish(std_msgs.msg.Bool(data=True))
            return

        # Reduce the veolicty when the robot is reaching the target.
        goal_distance = self.distance(self.position, self.goal)
        if goal_distance - self.reached_distance < self.slow_down_distance:
            speed = self.min_navigation_speed
            #Set the target as local target.
            if len(self.local_path) >= 2:
                self.local_path[-2]=self.global2local(self.goal)
        else:
            speed = self.max_navigation_speed

        #Variable initialization
        min_error = 1000.0
        best_command_found = False
        local_target = self.get_current_local_target()

        # Check possible action control commands
        for steer in np.arange(-MAX_STEER_ANGLE, MAX_STEER_ANGLE+0.01, self.delta_angle):
            if abs(steer)<0.01:
                steer=0.0
            #Reduce the velocity when the turn is big.
            k_sp = (MAX_STEER_ANGLE-abs(steer))/MAX_STEER_ANGLE
            speed2 = max(speed*k_sp, self.min_navigation_speed)

            # Check forwards movement. Reverse is reserved for the explicit
            # obstacle-recovery state machine, so normal navigation does not keep
            # driving backwards unnecessarily.
            directions = [speed2]
            for direction_speed in directions:
                flag_collision_risk = False
                sampled_trajectory = []
                minimum_obstacle_clearance = 10.0

                #TODO Calculate turn radius and velocity using the robot kinematics.
                angular_velocity = self.calculate_bicycle_angular_velocity(direction_speed, steer)

                # Calculate the trajectory using the angle rotation. Several points are sampled from the trajectory over time,
                # therefore the sample variable is equivalent to the t variable in the robot kinematic equation.
                for sample in np.arange(self.delta_sample, self.max_sample+0.01, self.delta_sample):
                    local_point=geometry_msgs.msg.Point()
                    #TODO calculate trajectory points using the robot kinematics.
                    local_point.x, local_point.y, final_heading = self.calculate_bicycle_trajectory_point(
                        direction_speed,
                        steer,
                        angular_velocity,
                        sample
                    )
                    sampled_trajectory.append((local_point, final_heading))

                    #TODO Detect collisions risk.
                    for obstacle in self.obstacles:
                        obstacle_distance = self.distance(local_point, obstacle)
                        if obstacle_distance < minimum_obstacle_clearance:
                            minimum_obstacle_clearance = obstacle_distance

                        if obstacle_distance < self.collision_margin:
                            flag_collision_risk = True
                            break

                    if flag_collision_risk:
                        break

                # If there is no collision, evaluate the trajectory.
                if not flag_collision_risk and len(sampled_trajectory) > 0:
                    #TODO estimate the trajectory evaluation in terms of the distance and orientation error to the local target (self.local_path[-2]).
                    final_point, final_heading = sampled_trajectory[-1]
                    distance_error = self.distance(final_point, local_target)
                    desired_heading = atan2(local_target.y - final_point.y, local_target.x - final_point.x)
                    orientation_error = abs(self.normalize_angle(desired_heading - final_heading))
                    steering_penalty = 0.12 * abs(steer) / MAX_STEER_ANGLE
                    clearance_reward = 0.05 * min(minimum_obstacle_clearance, 2.0)

                    error = 1.60 * distance_error + 0.90 * orientation_error + steering_penalty - clearance_reward
                    if error < min_error:
                        min_error=error
                        ackermann_control.steering_angle=steer
                        ackermann_control.speed=direction_speed
                        best_command_found = True

        if not best_command_found:
            rospy.logwarn_throttle(1.0, "No safe forward trajectory found. Starting local recovery.")
            self.start_recovery_state_machine(None)
            ackermann_control.speed = 0.0
            ackermann_control.steering_angle = 0.0

        #Publish message
        self.ackermann_command_publisher.publish(ackermann_control)

    def get_current_local_target(self):
        # -------------------------------------------------------------------------
        # BLOCK 6: Local target selection for trajectory scoring.
        # The original template evaluates against self.local_path[-2]. This helper
        # keeps that behaviour when the local planner has already produced points.
        # -------------------------------------------------------------------------
        if self.local_path is not None and len(self.local_path) >= 2:
            return self.local_path[-2]
        return self.global2local(self.goal)

    def calculate_bicycle_angular_velocity(self, speed, steering_angle):
        # -------------------------------------------------------------------------
        # BLOCK 7: Bicycle-model angular velocity.
        # omega = v / L * tan(delta). For near-zero steering, the robot moves in a
        # straight line and omega is set to zero to avoid numerical instability.
        # -------------------------------------------------------------------------
        if abs(steering_angle) < 1e-4:
            return 0.0
        return speed * tan(steering_angle) / VEHICLE_LENGHT

    def calculate_bicycle_trajectory_point(self, speed, steering_angle, angular_velocity, time_sample):
        # -------------------------------------------------------------------------
        # BLOCK 8: Bicycle-model trajectory sample.
        # The robot starts at local pose (0,0,0). For non-zero steering, an exact
        # circular-arc integration is used. For zero steering, the trajectory is a
        # straight line.
        # -------------------------------------------------------------------------
        if abs(angular_velocity) < 1e-4:
            x_position = speed * time_sample
            y_position = 0.0
            heading = 0.0
            return x_position, y_position, heading

        turn_radius = speed / angular_velocity
        heading = angular_velocity * time_sample
        x_position = turn_radius * sin(heading)
        y_position = turn_radius * (1.0 - cos(heading))

        return x_position, y_position, heading

    def detect_close_frontal_obstacle(self):
        # -------------------------------------------------------------------------
        # BLOCK 9: Close frontal obstacle detection.
        # The recovery behaviour is triggered only when an obstacle is near the
        # vehicle front. This prevents exaggerated early avoidance when the obstacle
        # is still far away.
        # -------------------------------------------------------------------------
        nearest_obstacle = None
        nearest_distance = 10000.0

        for obstacle in self.obstacles:
            if obstacle.x < self.front_obstacle_min_distance:
                continue
            if obstacle.x > self.front_obstacle_stop_distance:
                continue

            lateral_limit = self.front_obstacle_half_width + 0.12 * obstacle.x
            if abs(obstacle.y) > lateral_limit:
                continue

            obstacle_distance = sqrt(obstacle.x * obstacle.x + obstacle.y * obstacle.y)
            if obstacle_distance < nearest_distance:
                nearest_distance = obstacle_distance
                nearest_obstacle = obstacle

        return nearest_obstacle is not None, nearest_obstacle

    def start_recovery_state_machine(self, nearest_obstacle):
        # -------------------------------------------------------------------------
        # BLOCK 10: Start local obstacle recovery.
        # If the obstacle is on one side, the robot turns away from it. If the
        # obstacle is centered, the local target decides the preferred side. For the
        # requested test, the default centered behaviour turns right, keeping the
        # manoeuvre compact.
        # -------------------------------------------------------------------------
        local_target = self.get_current_local_target()

        if nearest_obstacle is not None and abs(nearest_obstacle.y) > 0.05:
            if nearest_obstacle.y > 0.0:
                self.recovery_turn_sign = -1.0
            else:
                self.recovery_turn_sign = 1.0
        else:
            if local_target.y > 0.05:
                self.recovery_turn_sign = 1.0
            else:
                self.recovery_turn_sign = -1.0

        self.recovery_state = "EMERGENCY_STOP"
        self.recovery_state_start_time = rospy.Time.now()

        rospy.logwarn(
            "Close obstacle detected. Starting recovery: stop -> reverse turn -> forward. turn_sign=%.1f",
            self.recovery_turn_sign
        )

    def execute_recovery_state_machine(self):
        # -------------------------------------------------------------------------
        # BLOCK 11: Execute local obstacle recovery.
        # Required behaviour: stop, reverse while steering slightly, then move
        # forward. If the frontal obstacle is still present at the end, repeat.
        # -------------------------------------------------------------------------
        elapsed = (rospy.Time.now() - self.recovery_state_start_time).to_sec()

        if self.recovery_state == "EMERGENCY_STOP":
            self.publish_ackermann_command(0.0, 0.0)
            if elapsed >= self.emergency_stop_duration:
                self.recovery_state = "REVERSE_TURN"
                self.recovery_state_start_time = rospy.Time.now()
            return

        if self.recovery_state == "REVERSE_TURN":
            reverse_steer = -self.recovery_turn_sign * self.recovery_steering_angle
            self.publish_ackermann_command(-self.recovery_reverse_speed, reverse_steer)
            if elapsed >= self.reverse_recovery_duration:
                self.recovery_state = "FORWARD_RECOVERY"
                self.recovery_state_start_time = rospy.Time.now()
            return

        if self.recovery_state == "FORWARD_RECOVERY":
            forward_steer = self.recovery_turn_sign * self.recovery_steering_angle
            self.publish_ackermann_command(self.recovery_forward_speed, forward_steer)
            if elapsed >= self.forward_recovery_duration:
                obstacle_found, nearest_obstacle = self.detect_close_frontal_obstacle()
                if obstacle_found:
                    rospy.logwarn("Obstacle still close after recovery. Repeating recovery sequence.")
                    self.start_recovery_state_machine(nearest_obstacle)
                else:
                    rospy.loginfo("Obstacle recovery completed. Returning to normal navigation.")
                    self.recovery_state = "NORMAL"
            return

        self.recovery_state = "NORMAL"

    def publish_ackermann_command(self, speed, steering_angle):
        command = ackermann_msgs.msg.AckermannDrive()
        command.speed = speed
        command.steering_angle = steering_angle
        self.ackermann_command_publisher.publish(command)

    def publish_stop_command(self):
        self.publish_ackermann_command(0.0, 0.0)

    def normalize_angle(self, angle_value):
        while angle_value > pi:
            angle_value -= 2.0 * pi
        while angle_value < -pi:
            angle_value += 2.0 * pi
        return angle_value

    def distance(self, p1, p2):
        return sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2)

    # Angle from point p2 to p1
    def angle(self, p1, p2):
        return atan2((p1.y-p2.y),p1.x-p2.x)

    #Transformation from global to local position
    def global2local(self, p):
        result = geometry_msgs.msg.Point()
        if self.position is None:
            return result
        #Translation
        x = (p.x-self.position.x)
        y = (p.y-self.position.y)
        #Rotation
        result.x = x * cos(-self.theta) - y * sin(-self.theta)
        result.y = x * sin(-self.theta) + y * cos(-self.theta)
        return result

    #From spherical to cartesian coordinates
    def spherical2Cartesian(self, depth, azimuth):
        sin_azimuth = sin(azimuth)
        cos_azimuth = cos(azimuth)
        p=geometry_msgs.msg.Point()
        p.x = depth * cos_azimuth
        p.y = depth * sin_azimuth
        return p

    #From cartesian to spherical coordinates
    def cartesian2Spherical(self, x,  y):
        depth = sqrt((x * x) + (y * y))

        azimuth = atan2(y, x)

        if (azimuth < 0): azimuth += 2*pi
        if (azimuth >= 2*pi): azimuth -= 2*pi

        return depth, azimuth

    def run(self):
        #Control loop
        rate = rospy.Rate(self.rate)
        count=3
        while not rospy.is_shutdown():
            self.controlActionCalculation()
            if count==0:
                self.localGoalCalculation()
                count=3
            else: count-=1
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
