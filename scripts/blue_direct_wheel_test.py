#!/usr/bin/env python3

# -------------------------------------------------------------------------
# BLOCK 1: Direct low-level wheel test.
# This script publishes directly to the BLUE wheel and steering controllers
# to verify that Gazebo can physically move the robot.
# -------------------------------------------------------------------------

import rospy
from std_msgs.msg import Float64


def publish_test_commands():
    rospy.init_node("blue_direct_wheel_test", anonymous=True)

    left_steering_pub = rospy.Publisher("/blue/left_steering_ctrlr/command", Float64, queue_size=1)
    right_steering_pub = rospy.Publisher("/blue/right_steering_ctrlr/command", Float64, queue_size=1)

    left_front_axle_pub = rospy.Publisher("/blue/left_front_axle_ctrlr/command", Float64, queue_size=1)
    right_front_axle_pub = rospy.Publisher("/blue/right_front_axle_ctrlr/command", Float64, queue_size=1)
    left_rear_axle_pub = rospy.Publisher("/blue/left_rear_axle_ctrlr/command", Float64, queue_size=1)
    right_rear_axle_pub = rospy.Publisher("/blue/right_rear_axle_ctrlr/command", Float64, queue_size=1)

    rospy.sleep(1.0)

    rate = rospy.Rate(20)

    # -------------------------------------------------------------------------
    # BLOCK 2: Move forward for 5 seconds.
    # If the robot moves backwards, change wheel_speed to -8.0.
    # -------------------------------------------------------------------------

    wheel_speed = 8.0
    steering_angle = 0.0

    start_time = rospy.Time.now()

    while not rospy.is_shutdown() and (rospy.Time.now() - start_time).to_sec() < 5.0:
        left_steering_pub.publish(Float64(data=steering_angle))
        right_steering_pub.publish(Float64(data=steering_angle))

        left_front_axle_pub.publish(Float64(data=wheel_speed))
        right_front_axle_pub.publish(Float64(data=wheel_speed))
        left_rear_axle_pub.publish(Float64(data=wheel_speed))
        right_rear_axle_pub.publish(Float64(data=wheel_speed))

        rate.sleep()

    # -------------------------------------------------------------------------
    # BLOCK 3: Stop all wheels.
    # -------------------------------------------------------------------------

    for _ in range(20):
        left_front_axle_pub.publish(Float64(data=0.0))
        right_front_axle_pub.publish(Float64(data=0.0))
        left_rear_axle_pub.publish(Float64(data=0.0))
        right_rear_axle_pub.publish(Float64(data=0.0))
        rate.sleep()


if __name__ == "__main__":
    try:
        publish_test_commands()
    except rospy.ROSInterruptException:
        pass
