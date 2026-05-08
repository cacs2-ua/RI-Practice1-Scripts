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
        self.reached_distance=1.5
        self.slow_down_distance=2.0
        self.rate=5

        #TODO Define other parameters needed for the planner.

        #Variable initialization
        self.position = None
        self.theta = 0 
        self.obstacles = []
        self.limits = None
        self.goal_reached = False #Wait until UR5 tells us to approach.
        #TODO Define other necessary variables.

        #TODO Target position initialization. It is possible to consider several target points to maneuver and approach the UR5 robot.
        self.goal = geometry_msgs.msg.Point()
        self.goal.x, self.goal.y = 0.0, 0.0
        # Local path initialization (4 points) to avoid errors until the first point is calculated.
        self.local_path=[self.goal for i in range(4)]
        print("Goal x: {}, y: {}".format(self.goal.x, self.goal.y))

        # TODO consider more subscribers/publishers if needed
        # Subscribers definition
        self.position_subscriber = rospy.Subscriber("/blue/ground_truth",
            Odometry, self.position_callback, queue_size=1)
        self.obstacles_subscriber = rospy.Subscriber("/obstacles",
            sensor_msgs.msg.PointCloud2, self.obstacles_callback, queue_size=1)
        self.obstacles_subscriber = rospy.Subscriber("/free_zone",
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

        
    #Callbacks
    #TODO modify this callback to not depend on the position given by Gazebo.
    def position_callback(self, ground_truth:Odometry):
        self.position=ground_truth.pose.pose.position
        euler = tf_conversions.transformations.euler_from_quaternion(ground_truth.pose.pose.orientation)
        self.theta=euler[2]

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
            for limit_point in self.limits:
                depth, azimuth=self.cartesian2Spherical(limit_point.x, limit_point.y)
                p_in  = self. spherical2Cartesian(rad, azimuth)

                distance_a=self.distance(p_in, self.local_path[0])
                if distance_a<0.0: distance_a=0.1
                force_a = wa2 /pow(distance_a,aa2)

                min_distance = 10000.0
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

    # Calculate the control commands to reach the planned local target.
    def controlActionCalculation(self):
        # Detect if the trajectory is finished.
        if self.position is None or self.goal_reached:
            return
        
        ackermann_control=ackermann_msgs.msg.AckermannDrive()
        ackermann_control.speed, ackermann_control.steering_angle = 0.0, 0.0
        # Detect if the robot has reached the following target point.
        if self.distance(self.position, self.goal) < self.reached_distance:
            print("Goal reached")
            self.goal_reached=True
            self.ackermann_command_publisher.publish(ackermann_control)
            return
        
        # Reduce the veolicty when the robot is reaching the target.
        goal_distance = self.distance(self.position, self.goal) 
        if goal_distance- self.reached_distance < self.slow_down_distance:
            speed = MIN_SPEED
            #Set the target as local target.
            self.local_path[-2]=self.global2local(self.goal)
        else:
            self.margin=1.0
            speed=MAX_SPEED

        #Variable initialization
        min_error = 1000

        # Check possible action control commands
        for steer in np.arange(-MAX_STEER_ANGLE, MAX_STEER_ANGLE+0.01, self.delta_angle):
            if abs(steer)<0.01: steer=0.0
            #Reduce the velocity when the turn is big.
            k_sp = (MAX_STEER_ANGLE-abs(steer))/MAX_STEER_ANGLE
            speed2 = max(speed*k_sp, MIN_SPEED)

            # Check forwards and backwards movement.
            directions = [-speed2, speed2]
            for dir in directions:
                flag_collision_risk = False

                #TODO Calculate turn radius and velocity using the robot kinematics.

                # Calculate the trajectory using the angle rotation. Several points are sampled from the trajectory over time, 
                # therefore the sample variable is equivalent to the t variable in the robot kinematic equation.
                for sample in np.arange(self.delta_sample, self.max_sample+0.01, self.delta_sample):
                    local_point=geometry_msgs.msg.Point()
                    #TODO calculate trajectory points using the robot kinematics.
                
                    #TODO Detect collisions risk.
                    for obstacle in self.obstacles:
                        pass
                    
                    if flag_collision_risk: break
 
                # If there is no collision, evaluate the trajectory.
                if not flag_collision_risk:
                    #TODO estimate the trajectory evaluation in terms of the distance and orientation error to the local target (self.local_path[-2]).
                    error = 0
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
    
    
