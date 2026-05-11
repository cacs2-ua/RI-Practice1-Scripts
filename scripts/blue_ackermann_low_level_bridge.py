#!/usr/bin/env python3
#
# Intelligent Robotics - Master's Degree in Artificial Intelligence - University of Alicante
#
# Auxiliary bridge for the BLUE Ackermann mobile robot.
#
# This node subscribes to:
#   /blue/ackermann_cmd              ackermann_msgs/AckermannDrive
#
# And publishes low-level Gazebo controller commands to:
#   /blue/left_rear_axle_ctrlr/command
#   /blue/right_rear_axle_ctrlr/command
#   /blue/left_steering_ctrlr/command
#   /blue/right_steering_ctrlr/command
#
# The practice planner outputs linear speed in m/s and steering angle in rad.
# Gazebo wheel velocity controllers usually expect wheel angular velocity in rad/s.
# Therefore:
#   wheel_angular_velocity = linear_speed / wheel_radius

import rospy
import ackermann_msgs.msg
import std_msgs.msg
from math import fabs


# This prevents sending unsafe values to the Gazebo low-level controllers.

def clamp(value, minimum_value, maximum_value):
    return max(min(value, maximum_value), minimum_value)


class BlueAckermannLowLevelBridge(object):

    def __init__(self):
        # This node is intentionally separated from blue_navigation.py so that the
        # planner remains focused on trajectory generation and this bridge remains
        # focused on low-level actuation.
        rospy.init_node("blue_ackermann_low_level_bridge", anonymous=True)

        # These parameters can be tuned from rosrun without editing the file.
        # If the car moves backwards, set _wheel_speed_sign:=-1.0.
        # If the steering is inverted, set _steering_angle_sign:=-1.0.
        self.wheel_radius = rospy.get_param("~wheel_radius", 0.30)
        self.wheel_speed_sign = rospy.get_param("~wheel_speed_sign", 1.0)
        self.steering_angle_sign = rospy.get_param("~steering_angle_sign", 1.0)

        self.max_wheel_angular_velocity = rospy.get_param("~max_wheel_angular_velocity", 5.80)
        self.max_steering_angle = rospy.get_param("~max_steering_angle", 0.42)

        self.command_timeout = rospy.get_param("~command_timeout", 0.60)
        self.publish_rate = rospy.get_param("~publish_rate", 20.0)

        self.ackermann_command_topic = rospy.get_param(
            "~ackermann_command_topic",
            "/blue/ackermann_cmd"
        )

        self.left_rear_wheel_topic = rospy.get_param(
            "~left_rear_wheel_topic",
            "/blue/left_rear_axle_ctrlr/command"
        )

        self.right_rear_wheel_topic = rospy.get_param(
            "~right_rear_wheel_topic",
            "/blue/right_rear_axle_ctrlr/command"
        )

        self.left_steering_topic = rospy.get_param(
            "~left_steering_topic",
            "/blue/left_steering_ctrlr/command"
        )

        self.right_steering_topic = rospy.get_param(
            "~right_steering_topic",
            "/blue/right_steering_ctrlr/command"
        )

        # The latest Ackermann command is stored and continuously republished to
        # the low-level controllers. If commands stop arriving, the robot stops.
        self.last_command_time = rospy.Time(0)
        self.target_wheel_angular_velocity = 0.0
        self.target_steering_angle = 0.0

        # These topics should be consumed by Gazebo velocity/position controllers.
        self.left_rear_wheel_publisher = rospy.Publisher(
            self.left_rear_wheel_topic,
            std_msgs.msg.Float64,
            queue_size=10
        )

        self.right_rear_wheel_publisher = rospy.Publisher(
            self.right_rear_wheel_topic,
            std_msgs.msg.Float64,
            queue_size=10
        )

        self.left_steering_publisher = rospy.Publisher(
            self.left_steering_topic,
            std_msgs.msg.Float64,
            queue_size=10
        )

        self.right_steering_publisher = rospy.Publisher(
            self.right_steering_topic,
            std_msgs.msg.Float64,
            queue_size=10
        )

        # This is the missing subscriber that must appear in:
        #   rostopic info /blue/ackermann_cmd
        self.ackermann_command_subscriber = rospy.Subscriber(
            self.ackermann_command_topic,
            ackermann_msgs.msg.AckermannDrive,
            self.ackermann_command_callback,
            queue_size=10
        )

        rospy.loginfo("BLUE Ackermann low-level bridge started.")
        rospy.loginfo("Subscribing to Ackermann command topic: %s", self.ackermann_command_topic)
        rospy.loginfo("Publishing left rear wheel command to: %s", self.left_rear_wheel_topic)
        rospy.loginfo("Publishing right rear wheel command to: %s", self.right_rear_wheel_topic)
        rospy.loginfo("Publishing left steering command to: %s", self.left_steering_topic)
        rospy.loginfo("Publishing right steering command to: %s", self.right_steering_topic)

        rospy.loginfo("wheel_radius=%.3f", self.wheel_radius)
        rospy.loginfo("wheel_speed_sign=%.1f", self.wheel_speed_sign)
        rospy.loginfo("steering_angle_sign=%.1f", self.steering_angle_sign)
        rospy.loginfo("max_wheel_angular_velocity=%.3f", self.max_wheel_angular_velocity)
        rospy.loginfo("max_steering_angle=%.3f", self.max_steering_angle)
        rospy.loginfo("command_timeout=%.3f", self.command_timeout)


    def ackermann_command_callback(self, ackermann_command):
        # The planner sends:
        #   speed in m/s
        #   steering_angle in rad
        #
        # This bridge converts speed into wheel angular velocity:
        #   rad/s = m/s / wheel_radius
        if self.wheel_radius <= 0.0:
            rospy.logwarn_throttle(1.0, "Invalid wheel radius. Stopping robot.")
            self.target_wheel_angular_velocity = 0.0
            self.target_steering_angle = 0.0
            return

        wheel_angular_velocity = ackermann_command.speed / self.wheel_radius
        wheel_angular_velocity *= self.wheel_speed_sign

        steering_angle = ackermann_command.steering_angle
        steering_angle *= self.steering_angle_sign

        wheel_angular_velocity = clamp(
            wheel_angular_velocity,
            -self.max_wheel_angular_velocity,
            self.max_wheel_angular_velocity
        )

        steering_angle = clamp(
            steering_angle,
            -self.max_steering_angle,
            self.max_steering_angle
        )

        self.target_wheel_angular_velocity = wheel_angular_velocity
        self.target_steering_angle = steering_angle
        self.last_command_time = rospy.Time.now()


    def publish_low_level_commands(self):
        # If the planner stops publishing commands, the bridge sends zero velocity
        # and zero steering angle after command_timeout seconds.
        current_time = rospy.Time.now()
        time_since_last_command = (current_time - self.last_command_time).to_sec()

        if time_since_last_command > self.command_timeout:
            wheel_angular_velocity = 0.0
            steering_angle = 0.0
        else:
            wheel_angular_velocity = self.target_wheel_angular_velocity
            steering_angle = self.target_steering_angle

        wheel_message = std_msgs.msg.Float64()
        wheel_message.data = wheel_angular_velocity

        steering_message = std_msgs.msg.Float64()
        steering_message.data = steering_angle

        self.left_rear_wheel_publisher.publish(wheel_message)
        self.right_rear_wheel_publisher.publish(wheel_message)

        self.left_steering_publisher.publish(steering_message)
        self.right_steering_publisher.publish(steering_message)


    def run(self):
        # The bridge continuously republishes the latest safe low-level command.
        rate = rospy.Rate(self.publish_rate)

        while not rospy.is_shutdown():
            self.publish_low_level_commands()
            rate.sleep()


def main():
    try:
        bridge = BlueAckermannLowLevelBridge()
        bridge.run()

    except rospy.ROSInterruptException:
        return

    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
