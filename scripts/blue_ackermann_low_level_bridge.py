#!/usr/bin/env python3

# -------------------------------------------------------------------------
# BLOCK 1: Ackermann-to-low-level BLUE bridge.
# This node subscribes to /blue/ackermann_cmd and converts AckermannDrive
# commands into the Float64 commands required by the actual Gazebo wheel and
# steering controllers.
# -------------------------------------------------------------------------

import rospy
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float64


class BlueAckermannLowLevelBridge(object):

    def __init__(self):
        rospy.init_node("blue_ackermann_low_level_bridge", anonymous=False)

        # -------------------------------------------------------------------------
        # BLOCK 2: Runtime parameters.
        # wheel_speed_sign is 1.0 because a positive low-level wheel speed moves
        # the vehicle forward in your direct tests. Change steering_angle_sign only
        # if left/right steering is inverted in your simulator.
        # -------------------------------------------------------------------------

        self.wheel_radius = rospy.get_param("~wheel_radius", 0.30)
        self.wheel_speed_sign = rospy.get_param("~wheel_speed_sign", 1.0)
        self.steering_angle_sign = rospy.get_param("~steering_angle_sign", 1.0)

        self.max_wheel_angular_velocity = rospy.get_param(
            "~max_wheel_angular_velocity",
            5.5
        )

        self.max_steering_angle = rospy.get_param(
            "~max_steering_angle",
            0.42
        )

        self.command_timeout = rospy.get_param(
            "~command_timeout",
            0.60
        )

        self.last_command_time = None

        # -------------------------------------------------------------------------
        # BLOCK 3: Low-level actuator publishers.
        # These are the Gazebo controller topics that actually move the BLUE robot.
        # -------------------------------------------------------------------------

        self.axle_publishers = [
            rospy.Publisher("/blue/left_front_axle_ctrlr/command", Float64, queue_size=10),
            rospy.Publisher("/blue/right_front_axle_ctrlr/command", Float64, queue_size=10),
            rospy.Publisher("/blue/left_rear_axle_ctrlr/command", Float64, queue_size=10),
            rospy.Publisher("/blue/right_rear_axle_ctrlr/command", Float64, queue_size=10),
        ]

        self.steering_publishers = [
            rospy.Publisher("/blue/left_steering_ctrlr/command", Float64, queue_size=10),
            rospy.Publisher("/blue/right_steering_ctrlr/command", Float64, queue_size=10),
        ]

        # -------------------------------------------------------------------------
        # BLOCK 4: Ackermann subscriber.
        # This makes /blue/ackermann_cmd have a real low-level subscriber.
        # -------------------------------------------------------------------------

        self.ackermann_subscriber = rospy.Subscriber(
            "/blue/ackermann_cmd",
            AckermannDrive,
            self.ackermann_callback,
            queue_size=10
        )

        rospy.on_shutdown(self.stop_robot)

        rospy.loginfo("BLUE Ackermann low-level bridge started.")
        rospy.loginfo("wheel_radius=%.3f", self.wheel_radius)
        rospy.loginfo("wheel_speed_sign=%.1f", self.wheel_speed_sign)
        rospy.loginfo("steering_angle_sign=%.1f", self.steering_angle_sign)

    def clamp(self, value, min_value, max_value):
        return max(min_value, min(max_value, value))

    def ackermann_callback(self, command):
        # -------------------------------------------------------------------------
        # BLOCK 5: Command conversion.
        # AckermannDrive.speed is interpreted as linear velocity in m/s.
        # Wheel controllers expect angular velocity in rad/s:
        #     wheel_angular_velocity = linear_speed / wheel_radius
        # -------------------------------------------------------------------------

        self.last_command_time = rospy.Time.now()

        if self.wheel_radius <= 0.0:
            rospy.logwarn_throttle(2.0, "Invalid wheel radius. Command ignored.")
            return

        wheel_angular_velocity = (
            self.wheel_speed_sign * command.speed / self.wheel_radius
        )

        steering_angle = (
            self.steering_angle_sign * command.steering_angle
        )

        wheel_angular_velocity = self.clamp(
            wheel_angular_velocity,
            -self.max_wheel_angular_velocity,
            self.max_wheel_angular_velocity
        )

        steering_angle = self.clamp(
            steering_angle,
            -self.max_steering_angle,
            self.max_steering_angle
        )

        self.publish_low_level_command(wheel_angular_velocity, steering_angle)

    def publish_low_level_command(self, wheel_angular_velocity, steering_angle):
        # -------------------------------------------------------------------------
        # BLOCK 6: Low-level publication.
        # The same wheel speed is sent to the four axle controllers. The same
        # steering angle is sent to both front steering controllers.
        # -------------------------------------------------------------------------

        wheel_message = Float64()
        wheel_message.data = wheel_angular_velocity

        steering_message = Float64()
        steering_message.data = steering_angle

        for publisher in self.axle_publishers:
            publisher.publish(wheel_message)

        for publisher in self.steering_publishers:
            publisher.publish(steering_message)

    def stop_robot(self):
        # -------------------------------------------------------------------------
        # BLOCK 7: Safety stop.
        # Sends zero velocity and zero steering when the node finishes.
        # -------------------------------------------------------------------------

        self.publish_low_level_command(0.0, 0.0)

    def run(self):
        # -------------------------------------------------------------------------
        # BLOCK 8: Watchdog loop.
        # If the planner stops publishing commands, the bridge stops the robot.
        # -------------------------------------------------------------------------

        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            if self.last_command_time is not None:
                elapsed = (rospy.Time.now() - self.last_command_time).to_sec()

                if elapsed > self.command_timeout:
                    self.stop_robot()

            rate.sleep()


def main():
    bridge = BlueAckermannLowLevelBridge()
    bridge.run()


if __name__ == "__main__":
    main()
