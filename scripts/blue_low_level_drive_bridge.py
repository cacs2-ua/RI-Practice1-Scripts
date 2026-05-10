#!/usr/bin/env python3

# Intelligent Robotics - Master's Degree in Artificial Intelligence - University of Alicante

# -------------------------------------------------------------------------
# BLOCK 1: Low-level compatibility bridge.
# The practice requires blue_navigation.py to publish AckermannDrive commands
# to /blue/ackermann_cmd. In this Docker/simulation instance, the default
# /blue/ackermann_controller is not subscribed to that topic. This bridge keeps
# the required high-level interface and forwards the command to Gazebo's low-
# level steering and axle controllers.
# -------------------------------------------------------------------------

import rospy
from math import pi, tan, atan
from std_msgs.msg import Float64
from ackermann_msgs.msg import AckermannDrive


MAX_STEERING_ANGLE = 24.0 * pi / 180.0
VEHICLE_LENGTH = 1.05
FRONT_TRACK_WIDTH = 0.85
WHEEL_RADIUS = 0.16
COMMAND_TIMEOUT = 0.5


class BlueLowLevelDriveBridge(object):

    def __init__(self):
        rospy.init_node("blue_low_level_drive_bridge", anonymous=True)

        # -------------------------------------------------------------------------
        # BLOCK 2: Runtime parameters.
        # These parameters allow the bridge to be tuned without editing the file.
        # -------------------------------------------------------------------------

        self.vehicle_length = rospy.get_param("~vehicle_length", VEHICLE_LENGTH)
        self.front_track_width = rospy.get_param("~front_track_width", FRONT_TRACK_WIDTH)
        self.wheel_radius = rospy.get_param("~wheel_radius", WHEEL_RADIUS)
        self.wheel_speed_gain = rospy.get_param("~wheel_speed_gain", 1.0)
        self.invert_wheel_speed = rospy.get_param("~invert_wheel_speed", False)

        self.current_speed = 0.0
        self.current_steering_angle = 0.0
        self.last_command_time = rospy.Time.now()

        # -------------------------------------------------------------------------
        # BLOCK 3: Low-level steering publishers.
        # These topics are consumed by Gazebo.
        # -------------------------------------------------------------------------

        self.left_steering_publisher = rospy.Publisher(
            "/blue/left_steering_ctrlr/command",
            Float64,
            queue_size=1
        )

        self.right_steering_publisher = rospy.Publisher(
            "/blue/right_steering_ctrlr/command",
            Float64,
            queue_size=1
        )

        # -------------------------------------------------------------------------
        # BLOCK 4: Low-level axle publishers.
        # These topics are consumed by Gazebo and were verified with the direct
        # wheel test.
        # -------------------------------------------------------------------------

        self.left_front_axle_publisher = rospy.Publisher(
            "/blue/left_front_axle_ctrlr/command",
            Float64,
            queue_size=1
        )

        self.right_front_axle_publisher = rospy.Publisher(
            "/blue/right_front_axle_ctrlr/command",
            Float64,
            queue_size=1
        )

        self.left_rear_axle_publisher = rospy.Publisher(
            "/blue/left_rear_axle_ctrlr/command",
            Float64,
            queue_size=1
        )

        self.right_rear_axle_publisher = rospy.Publisher(
            "/blue/right_rear_axle_ctrlr/command",
            Float64,
            queue_size=1
        )

        # -------------------------------------------------------------------------
        # BLOCK 5: High-level Ackermann subscriber.
        # This creates a real subscriber for /blue/ackermann_cmd.
        # -------------------------------------------------------------------------

        self.ackermann_subscriber = rospy.Subscriber(
            "/blue/ackermann_cmd",
            AckermannDrive,
            self.ackermann_command_callback,
            queue_size=1
        )

        rospy.loginfo("BLUE low-level drive bridge started.")
        rospy.loginfo("Listening to /blue/ackermann_cmd.")
        rospy.loginfo("Publishing to BLUE low-level wheel and steering controllers.")
        rospy.loginfo("Wheel radius: %.3f", self.wheel_radius)
        rospy.loginfo("Wheel speed gain: %.3f", self.wheel_speed_gain)
        rospy.loginfo("Invert wheel speed: %s", str(self.invert_wheel_speed))

    def ackermann_command_callback(self, command_msg):
        # -------------------------------------------------------------------------
        # BLOCK 6: Store the latest Ackermann command.
        # speed is in m/s and steering_angle is in radians.
        # -------------------------------------------------------------------------

        self.current_speed = command_msg.speed
        self.current_steering_angle = self.clamp(
            command_msg.steering_angle,
            -MAX_STEERING_ANGLE,
            MAX_STEERING_ANGLE
        )
        self.last_command_time = rospy.Time.now()

    def clamp(self, value, min_value, max_value):
        return max(min(value, max_value), min_value)

    def compute_ackermann_steering(self, steering_angle):
        # -------------------------------------------------------------------------
        # BLOCK 7: Ackermann steering geometry.
        # The inner and outer front wheels receive different steering angles.
        # -------------------------------------------------------------------------

        if abs(steering_angle) < 0.001:
            return 0.0, 0.0

        turn_radius = self.vehicle_length / tan(steering_angle)

        left_angle = atan(
            self.vehicle_length / (turn_radius - self.front_track_width / 2.0)
        )

        right_angle = atan(
            self.vehicle_length / (turn_radius + self.front_track_width / 2.0)
        )

        left_angle = self.clamp(left_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
        right_angle = self.clamp(right_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

        return left_angle, right_angle

    def publish_low_level_commands(self):
        # -------------------------------------------------------------------------
        # BLOCK 8: Convert high-level commands into low-level controller commands.
        # If no Ackermann command is received recently, the robot is stopped.
        # -------------------------------------------------------------------------

        elapsed_time = (rospy.Time.now() - self.last_command_time).to_sec()

        if elapsed_time > COMMAND_TIMEOUT:
            speed = 0.0
            steering_angle = 0.0
        else:
            speed = self.current_speed
            steering_angle = self.current_steering_angle

        left_steering_angle, right_steering_angle = self.compute_ackermann_steering(
            steering_angle
        )

        wheel_angular_speed = self.wheel_speed_gain * speed / self.wheel_radius

        if self.invert_wheel_speed:
            wheel_angular_speed = -wheel_angular_speed

        self.left_steering_publisher.publish(Float64(data=left_steering_angle))
        self.right_steering_publisher.publish(Float64(data=right_steering_angle))

        self.left_front_axle_publisher.publish(Float64(data=wheel_angular_speed))
        self.right_front_axle_publisher.publish(Float64(data=wheel_angular_speed))
        self.left_rear_axle_publisher.publish(Float64(data=wheel_angular_speed))
        self.right_rear_axle_publisher.publish(Float64(data=wheel_angular_speed))

    def run(self):
        # -------------------------------------------------------------------------
        # BLOCK 9: Main bridge loop.
        # -------------------------------------------------------------------------

        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            self.publish_low_level_commands()
            rate.sleep()


def main():
    try:
        bridge = BlueLowLevelDriveBridge()
        bridge.run()
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()