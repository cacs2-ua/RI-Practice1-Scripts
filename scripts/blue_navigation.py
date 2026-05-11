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
MAX_SPEED = 1.3    # m/s
MIN_SPEED = 0.6
VEHICLE_LENGHT = 1.05 

class BlueTrajectoryPlanner(object):

    def __init__(self):
        super(BlueTrajectoryPlanner, self).__init__()

        ##ROS node initialization
        rospy.init_node("blue_planner_node", anonymous=True)

        #Parameter initialization
        self.delta_angle=6.0*pi/180.0
        self.delta_sample=0.2
        self.max_sample=1.0
        self.reached_distance=0.95
        self.slow_down_distance=2.0
        self.rate=5

        # -------------------------------------------------------------------------
        # BLOCK 1: Planner and recovery parameters.
        # These parameters keep the robot at a practical speed while avoiding very
        # wide detours. The recovery maneuver is only triggered when the obstacle is
        # close and in front of the vehicle.
        # -------------------------------------------------------------------------

        #TODO Define other parameters needed for the planner.
        self.collision_margin = rospy.get_param("~collision_margin", 0.42)
        self.clearance_weight = rospy.get_param("~clearance_weight", 0.10)
        self.distance_error_weight = rospy.get_param("~distance_error_weight", 1.00)
        self.heading_error_weight = rospy.get_param("~heading_error_weight", 0.75)
        self.steering_effort_weight = rospy.get_param("~steering_effort_weight", 0.12)
        self.reverse_motion_penalty = rospy.get_param("~reverse_motion_penalty", 4.0)

        self.nominal_navigation_speed = rospy.get_param("~nominal_navigation_speed", 1.10)
        self.minimum_navigation_speed = rospy.get_param("~minimum_navigation_speed", MIN_SPEED)

        self.near_obstacle_distance = rospy.get_param("~near_obstacle_distance", 1.05)
        self.front_corridor_half_width = rospy.get_param("~front_corridor_half_width", 0.48)
        self.front_obstacle_min_x = rospy.get_param("~front_obstacle_min_x", 0.25)

        self.stop_cycles_before_reverse = rospy.get_param("~stop_cycles_before_reverse", 2)
        self.reverse_cycles = rospy.get_param("~reverse_cycles", 6)
        self.forward_recovery_cycles = rospy.get_param("~forward_recovery_cycles", 7)
        self.recovery_reverse_speed = rospy.get_param("~recovery_reverse_speed", -0.65)
        self.recovery_forward_speed = rospy.get_param("~recovery_forward_speed", 0.80)
        self.recovery_steering_angle = rospy.get_param("~recovery_steering_angle", 0.30)

        self.use_camera_localization = rospy.get_param("~use_camera_localization", True)
        self.camera_localization_timeout = rospy.get_param("~camera_localization_timeout", 1.0)
        self.min_camera_motion_for_heading = rospy.get_param("~min_camera_motion_for_heading", 0.04)

        self.wait_for_ur5_signal = rospy.get_param("~wait_for_ur5_signal", False)
        self.navigation_enabled = not self.wait_for_ur5_signal

        #Variable initialization
        self.position = None
        self.theta = 0 
        self.obstacles = []
        self.limits = None
        self.goal_reached = False #Wait until UR5 tells us to approach.

        # -------------------------------------------------------------------------
        # BLOCK 2: Navigation state variables.
        # The normal planner follows the selected waypoint. The recovery state
        # machine performs stop -> reverse while steering -> forward when a close
        # obstacle is detected directly in front of the robot.
        # -------------------------------------------------------------------------

        #TODO Define other necessary variables.
        self.recovery_state = "NORMAL_NAVIGATION"
        self.recovery_cycle_counter = 0
        self.recovery_steering_sign = 1.0
        self.last_camera_position = None
        self.last_camera_update_time = None
        self.last_ground_truth_position = None

        # -------------------------------------------------------------------------
        # BLOCK 3: Target position initialization.
        # The final goal is the requested test goal. An intermediate waypoint makes
        # the vehicle approach the obstacle lane first, so the obstacle at
        # (3.10, -6.03) becomes a meaningful Part 2 navigation test.
        # -------------------------------------------------------------------------

        #TODO Target position initialization. It is possible to consider several target points to maneuver and approach the UR5 robot.
        final_goal_x = rospy.get_param("~goal_x", 6.60)
        final_goal_y = rospy.get_param("~goal_y", -6.03)
        use_intermediate_goal = rospy.get_param("~use_intermediate_goal", True)
        intermediate_goal_x = rospy.get_param("~intermediate_goal_x", 2.40)
        intermediate_goal_y = rospy.get_param("~intermediate_goal_y", -6.03)

        self.goal_waypoints = []

        if use_intermediate_goal:
            intermediate_goal = geometry_msgs.msg.Point()
            intermediate_goal.x = intermediate_goal_x
            intermediate_goal.y = intermediate_goal_y
            intermediate_goal.z = 0.0
            self.goal_waypoints.append(intermediate_goal)

        final_goal = geometry_msgs.msg.Point()
        final_goal.x = final_goal_x
        final_goal.y = final_goal_y
        final_goal.z = 0.0
        self.goal_waypoints.append(final_goal)

        self.current_goal_index = 0
        self.goal = self.goal_waypoints[self.current_goal_index]

        # Local path initialization (4 points) to avoid errors until the first point is calculated.
        self.local_path=[self.goal for i in range(4)]
        print("Goal x: {}, y: {}".format(self.goal.x, self.goal.y))

        # -------------------------------------------------------------------------
        # BLOCK 4: Subscribers and publishers.
        # Ground truth is kept as a fallback only. Camera localization from
        # /pose_array is used by default, satisfying the localization requirement.
        # -------------------------------------------------------------------------

        # TODO consider more subscribers/publishers if needed
        # Subscribers definition
        self.position_subscriber = rospy.Subscriber("/blue/ground_truth",
            Odometry, self.position_callback, queue_size=1)
        self.camera_position_subscriber = rospy.Subscriber("/pose_array",
            geometry_msgs.msg.PoseArray, self.camera_pose_callback, queue_size=1)
        self.obstacles_subscriber = rospy.Subscriber("/obstacles",
            sensor_msgs.msg.PointCloud2, self.obstacles_callback, queue_size=1)
        self.limits_subscriber = rospy.Subscriber("/free_zone",
            sensor_msgs.msg.PointCloud2, self.limits_callback, queue_size=1)
        self.start_navigation_subscriber = rospy.Subscriber("/blue/start_navigation",
            std_msgs.msg.Bool, self.start_navigation_callback, queue_size=1)

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
            queue_size=1,
            latch=True,
        )

        rospy.on_shutdown(self.publish_stop_command)

    #Callbacks
    #TODO modify this callback to not depend on the position given by Gazebo.
    def position_callback(self, ground_truth:Odometry):
        # -------------------------------------------------------------------------
        # BLOCK 5: Ground-truth fallback localization.
        # The planner does not depend on this callback when camera localization is
        # available. It is kept as a robust fallback for debugging.
        # -------------------------------------------------------------------------

        self.last_ground_truth_position = ground_truth.pose.pose.position

        if self.use_camera_localization and self.is_camera_localization_recent():
            return

        self.position = ground_truth.pose.pose.position

        # -------------------------------------------------------------------------
        # BLOCK 5.1: Quaternion conversion for tf.
        # tf_conversions.transformations.euler_from_quaternion expects a list or tuple,
        # not a geometry_msgs/Quaternion object.
        # -------------------------------------------------------------------------

        orientation = ground_truth.pose.pose.orientation

        quaternion_as_list = [
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ]

        euler = tf_conversions.transformations.euler_from_quaternion(quaternion_as_list)
        self.theta = euler[2]

    def camera_pose_callback(self, pose_array:geometry_msgs.msg.PoseArray):
        # -------------------------------------------------------------------------
        # BLOCK 6: Camera-based BLUE localization.
        # The object_localization node publishes the red object in poses[0] and the
        # BLUE robot in poses[1]. The robot heading is approximated from consecutive
        # camera positions, as requested in the localization part of the assignment.
        # -------------------------------------------------------------------------

        if not self.use_camera_localization:
            return

        if len(pose_array.poses) < 2:
            return

        detected_robot_position = pose_array.poses[1].position

        if detected_robot_position.z < 0.0:
            return

        camera_position = geometry_msgs.msg.Point()
        camera_position.x = detected_robot_position.x
        camera_position.y = detected_robot_position.y
        camera_position.z = 0.0

        if self.last_camera_position is not None:
            displacement = self.distance(camera_position, self.last_camera_position)

            if displacement >= self.min_camera_motion_for_heading:
                self.theta = atan2(
                    camera_position.y - self.last_camera_position.y,
                    camera_position.x - self.last_camera_position.x
                )

        self.position = camera_position
        self.last_camera_position = copy.deepcopy(camera_position)
        self.last_camera_update_time = rospy.Time.now()

    def start_navigation_callback(self, start_msg:std_msgs.msg.Bool):
        # -------------------------------------------------------------------------
        # BLOCK 7: Optional Part 3 synchronization input.
        # If wait_for_ur5_signal is true, the BLUE robot waits until the UR5 node
        # publishes True on /blue/start_navigation.
        # -------------------------------------------------------------------------

        self.navigation_enabled = bool(start_msg.data)

        if self.navigation_enabled:
            rospy.loginfo("BLUE navigation enabled by /blue/start_navigation.")

    def obstacles_callback(self, obstacles:sensor_msgs.msg.PointCloud2):
        pc_obstacles=pc2.read_points(obstacles, field_names=("x", "y", "z"), skip_nans=True)
        #Save as geometry_msg.Point
        self.obstacles=[]
        for point in pc_obstacles:
            x,y,z = point
            new_point = geometry_msgs.msg.Point(x,y,z)
            self.obstacles.append(new_point)

    def limits_callback(self, limits:sensor_msgs.msg.PointCloud2):
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

    def is_camera_localization_recent(self):
        # -------------------------------------------------------------------------
        # BLOCK 8: Camera localization freshness check.
        # This avoids using old camera detections if the camera node stops.
        # -------------------------------------------------------------------------

        if self.last_camera_update_time is None:
            return False

        elapsed = (rospy.Time.now() - self.last_camera_update_time).to_sec()
        return elapsed <= self.camera_localization_timeout

    def publish_stop_command(self):
        # -------------------------------------------------------------------------
        # BLOCK 9: Safety stop command.
        # -------------------------------------------------------------------------

        stop_command = ackermann_msgs.msg.AckermannDrive()
        stop_command.speed = 0.0
        stop_command.steering_angle = 0.0
        self.ackermann_command_publisher.publish(stop_command)

    # Trajectory planning towards the target while avoiding obstacles. It is executed at a lower frequency.
    ''' Code implemented from the following research article:
            OpenStreetMap-Based Autonomous Navigation With LiDAR Naive-Valley-Path Obstacle Avoidance
            Miguel Ángel Muñoz Bañón, Edison Velasco Sánchez, Francisco A. Candelas, Fernando Torres
            IEEE Transactions on Intelligent Transportation Systems, 2022
    '''
    def localGoalCalculation(self):
        if self.position is  None or self.goal_reached or self.limits is None: return
        #Parameter definition
        wa, wr, ar, aa = 100.0, 1.0, 0.8, 0.3
        min_force = wr / pow(0.1, ar) - wa / pow(100.0, aa) + 1000

        local_goal=geometry_msgs.msg.Point()
        self.local_path=[]
        min_distance = 10000.0
        goal_in_local_axis = self.global2local(self.goal)
        # 1) Trajectory points within the limited radius.
        for limit_point in self.limits:
            distance_a = self.distance(limit_point, goal_in_local_axis)
            if distance_a<0.0: distance_a=0.1

            force_a = wa / pow(distance_a,aa)

            #Computation of the minimum obstacle distance and its weight.
            min_distance=10000.0
            force_r = 0.0
            for obstacle_point in self.obstacles:
                distance_r = self.distance(limit_point,obstacle_point)
                if distance_r<0.1: distance_r=0.1
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
        radious = 4.0
        delta_rad = radious / 3.0

        for rad in np.arange(radious, delta_rad-0.1, -delta_rad):
            min_force = wr / pow(0.1, ar) - wa / pow(100.0, aa) + 1000
            #Computation of the minimum obstacle distance and its weight.
            min_distance = 10000.0
            local_goal = self.global2local(self.goal)
            for limit_point in self.limits:
                depth, azimuth=self.cartesian2Spherical(limit_point.x, limit_point.y)
                p_in  = self.spherical2Cartesian(rad, azimuth)

                distance_a=self.distance(p_in, self.local_path[0])
                if distance_a<0.0: distance_a=0.1
                force_a = wa2 /pow(distance_a,aa2)

                min_distance = 10000.0
                force_r = 0.0
                for obstacle_point in self.obstacles:
                    distance_r = self.distance(p_in, obstacle_point)
                    if distance_r<0.1: distance_r=0.1
                    if distance_r<min_distance:
                        min_distance=distance_r
                        force_r = wr2 / pow(distance_r, ar2)

                force = force_r-force_a
                if force < min_force:
                    min_force=force
                    local_goal=p_in
            self.local_path.append(local_goal)

        if len(self.local_path) < 2:
            self.local_path.append(goal_in_local_axis)

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
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker_msg.markers.append(marker)
        self.marker_publisher.publish(marker_msg)

    def get_front_obstacle(self):
        # -------------------------------------------------------------------------
        # BLOCK 10: Close frontal obstacle detection.
        # Obstacles are already expressed in the BLUE/Velodyne local frame. The
        # recovery maneuver starts only when a point is close, in front, and inside
        # a narrow corridor. This prevents exaggerated early detours.
        # -------------------------------------------------------------------------

        closest_obstacle = None
        closest_distance = 10000.0

        for obstacle in self.obstacles:
            if obstacle.x < self.front_obstacle_min_x:
                continue

            if fabs(obstacle.y) > self.front_corridor_half_width:
                continue

            obstacle_distance = sqrt(obstacle.x ** 2 + obstacle.y ** 2)

            if obstacle_distance < closest_distance:
                closest_distance = obstacle_distance
                closest_obstacle = obstacle

        if closest_obstacle is None:
            return None, closest_distance

        if closest_distance <= self.near_obstacle_distance:
            return closest_obstacle, closest_distance

        return None, closest_distance

    def start_recovery_maneuver(self, front_obstacle):
        # -------------------------------------------------------------------------
        # BLOCK 11: Recovery maneuver initialization.
        # The robot first stops, then reverses while steering slightly away from the
        # obstacle side, and finally moves forward again to rejoin the planned route.
        # -------------------------------------------------------------------------

        self.recovery_state = "STOP_BEFORE_OBSTACLE"
        self.recovery_cycle_counter = self.stop_cycles_before_reverse

        if front_obstacle is not None and front_obstacle.y >= 0.0:
            self.recovery_steering_sign = -1.0
        else:
            self.recovery_steering_sign = 1.0

        rospy.logwarn(
            "Close obstacle detected. Starting recovery: stop -> reverse with steering -> forward."
        )

    def execute_recovery_maneuver(self):
        # -------------------------------------------------------------------------
        # BLOCK 12: Stop / reverse / forward recovery command generation.
        # If the obstacle is still detected after the forward phase, the same process
        # is repeated, exactly as requested.
        # -------------------------------------------------------------------------

        recovery_command = ackermann_msgs.msg.AckermannDrive()
        recovery_command.speed = 0.0
        recovery_command.steering_angle = 0.0

        if self.recovery_state == "STOP_BEFORE_OBSTACLE":
            recovery_command.speed = 0.0
            recovery_command.steering_angle = 0.0
            self.recovery_cycle_counter -= 1

            if self.recovery_cycle_counter <= 0:
                self.recovery_state = "REVERSING_WITH_STEERING"
                self.recovery_cycle_counter = self.reverse_cycles

        elif self.recovery_state == "REVERSING_WITH_STEERING":
            recovery_command.speed = self.recovery_reverse_speed
            recovery_command.steering_angle = self.recovery_steering_sign * self.recovery_steering_angle
            self.recovery_cycle_counter -= 1

            if self.recovery_cycle_counter <= 0:
                self.recovery_state = "FORWARD_RECOVERY"
                self.recovery_cycle_counter = self.forward_recovery_cycles

        elif self.recovery_state == "FORWARD_RECOVERY":
            recovery_command.speed = self.recovery_forward_speed
            recovery_command.steering_angle = -self.recovery_steering_sign * self.recovery_steering_angle * 0.80
            self.recovery_cycle_counter -= 1

            if self.recovery_cycle_counter <= 0:
                front_obstacle, _ = self.get_front_obstacle()

                if front_obstacle is not None:
                    self.start_recovery_maneuver(front_obstacle)
                else:
                    self.recovery_state = "NORMAL_NAVIGATION"

        self.ackermann_command_publisher.publish(recovery_command)
        return self.recovery_state != "NORMAL_NAVIGATION"

    def update_current_goal_if_needed(self):
        # -------------------------------------------------------------------------
        # BLOCK 13: Waypoint management.
        # The robot first reaches the intermediate waypoint near the obstacle lane
        # and then continues to the requested final goal.
        # -------------------------------------------------------------------------

        if self.position is None:
            return

        if self.distance(self.position, self.goal) >= self.reached_distance:
            return

        if self.current_goal_index < len(self.goal_waypoints) - 1:
            self.current_goal_index += 1
            self.goal = self.goal_waypoints[self.current_goal_index]
            rospy.loginfo("Switching to next goal x=%.2f, y=%.2f", self.goal.x, self.goal.y)
            return

        print("Goal reached")
        self.goal_reached=True
        self.navigation_finished_publisher.publish(std_msgs.msg.Bool(data=True))
        self.publish_stop_command()

    # Calculate the control commands to reach the planned local target.
    def controlActionCalculation(self):
        # Detect if the trajectory is finished.
        if self.position is None or self.goal_reached:
            return

        if not self.navigation_enabled:
            self.publish_stop_command()
            return

        self.update_current_goal_if_needed()

        if self.goal_reached:
            return

        front_obstacle, front_obstacle_distance = self.get_front_obstacle()

        if self.recovery_state == "NORMAL_NAVIGATION" and front_obstacle is not None:
            self.start_recovery_maneuver(front_obstacle)

        if self.recovery_state != "NORMAL_NAVIGATION":
            self.execute_recovery_maneuver()
            return
        
        ackermann_control=ackermann_msgs.msg.AckermannDrive()
        ackermann_control.speed, ackermann_control.steering_angle = 0.0, 0.0

        # Reduce the veolicty when the robot is reaching the target.
        goal_distance = self.distance(self.position, self.goal) 
        if goal_distance- self.reached_distance < self.slow_down_distance:
            speed = self.minimum_navigation_speed
            #Set the target as local target.
            if len(self.local_path) >= 2:
                self.local_path[-2]=self.global2local(self.goal)
        else:
            self.margin=1.0
            speed=min(MAX_SPEED, self.nominal_navigation_speed)

        #Variable initialization
        min_error = 1000

        if len(self.local_path) < 2:
            local_target = self.global2local(self.goal)
        else:
            local_target = self.local_path[-2]

        # Check possible action control commands
        for steer in np.arange(-MAX_STEER_ANGLE, MAX_STEER_ANGLE+0.01, self.delta_angle):
            if abs(steer)<0.01: steer=0.0
            #Reduce the velocity when the turn is big.
            k_sp = (MAX_STEER_ANGLE-abs(steer))/MAX_STEER_ANGLE
            speed2 = max(speed*k_sp, self.minimum_navigation_speed)

            # Check forwards and backwards movement.
            directions = [speed2]
            for dir in directions:
                flag_collision_risk = False

                #TODO Calculate turn radius and velocity using the robot kinematics.
                if abs(steer) < 0.001:
                    turn_radius = 1000000.0
                    angular_velocity = 0.0
                else:
                    turn_radius = VEHICLE_LENGHT / tan(steer)
                    angular_velocity = dir / turn_radius

                final_point = geometry_msgs.msg.Point()
                final_heading = 0.0
                minimum_obstacle_clearance = 10000.0

                # Calculate the trajectory using the angle rotation. Several points are sampled from the trajectory over time, 
                # therefore the sample variable is equivalent to the t variable in the robot kinematic equation.
                for sample in np.arange(self.delta_sample, self.max_sample+0.01, self.delta_sample):
                    local_point=geometry_msgs.msg.Point()
                    #TODO calculate trajectory points using the robot kinematics.
                    if abs(angular_velocity) < 0.001:
                        local_point.x = dir * sample
                        local_point.y = 0.0
                        local_heading = 0.0
                    else:
                        local_heading = angular_velocity * sample
                        local_point.x = turn_radius * sin(local_heading)
                        local_point.y = turn_radius * (1.0 - cos(local_heading))

                    final_point = local_point
                    final_heading = local_heading
                
                    #TODO Detect collisions risk.
                    for obstacle in self.obstacles:
                        obstacle_distance = self.distance(local_point, obstacle)
                        minimum_obstacle_clearance = min(minimum_obstacle_clearance, obstacle_distance)

                        if obstacle_distance < self.collision_margin:
                            flag_collision_risk = True
                            break
                    
                    if flag_collision_risk: break
 
                # If there is no collision, evaluate the trajectory.
                if not flag_collision_risk:
                    #TODO estimate the trajectory evaluation in terms of the distance and orientation error to the local target (self.local_path[-2]).
                    distance_error = self.distance(final_point, local_target)
                    heading_to_target = self.angle(local_target, final_point)
                    heading_error = fabs(self.normalize_angle(heading_to_target - final_heading))
                    steering_error = fabs(steer) / MAX_STEER_ANGLE

                    clearance_error = 0.0
                    if minimum_obstacle_clearance < 10000.0:
                        clearance_error = 1.0 / max(minimum_obstacle_clearance, self.collision_margin)

                    reverse_error = self.reverse_motion_penalty if dir < 0.0 else 0.0

                    error = (
                        self.distance_error_weight * distance_error +
                        self.heading_error_weight * heading_error +
                        self.steering_effort_weight * steering_error +
                        self.clearance_weight * clearance_error +
                        reverse_error
                    )

                    if error < min_error:
                        min_error=error
                        ackermann_control.steering_angle=steer
                        ackermann_control.speed=dir
        
        #Publish message
        self.ackermann_command_publisher.publish(ackermann_control)
    
    
    def distance(self, p1:geometry_msgs.msg.Point, p2:geometry_msgs.msg.Point):
        return sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2)
    
    # Angle from point p2 to p1
    def angle(self, p1:geometry_msgs.msg.Point, p2:geometry_msgs.msg.Point):
        return atan2((p1.y-p2.y),p1.x-p2.x)

    def normalize_angle(self, angle_value):
        # -------------------------------------------------------------------------
        # BLOCK 14: Angle normalization.
        # -------------------------------------------------------------------------

        while angle_value > pi:
            angle_value -= 2.0 * pi

        while angle_value < -pi:
            angle_value += 2.0 * pi

        return angle_value
    
    #Transformation from global to local position
    def global2local(self, p:geometry_msgs.msg.Point):
        result = geometry_msgs.msg.Point()
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