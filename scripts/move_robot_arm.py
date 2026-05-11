#!/usr/bin/env python3
#
# ROS node to move the UR5 arm robot to the goal position using MoveIt and
# open/close the gripper. It comunicates with the other nodes through the topic
# /ur5_goal, the student should add the appropiate topics for coordinate the tasks.
#
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, SRI International
# All rights reserved.

from __future__ import print_function

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import std_msgs.msg
import tf
from tf.transformations import quaternion_from_euler

from math import pi, dist, cos, fabs
from moveit_commander.conversions import pose_to_list


# -------------------------------------------------------------------------
# BLOCK 1: Global configuration of the UR5 position and default joint states.
# ROBOT_POSITION represents the global reference position used to convert the
# RGB-D object localization into the UR5 planning reference.
# -------------------------------------------------------------------------

#TODO Define the global position of the robot.
ROBOT_POSITION = geometry_msgs.msg.Point(x=0.0, y=0.0, z=0.0)

#TODO Define joint positions of the arm (home position) and of the gripper (open and close)
# They muss be in radians.

# Safe home-like configuration for the UR5 arm.
HOME_JOINT_STATE = [
    0.0,
    -pi / 2.0,
    pi / 2.0,
    -pi / 2.0,
    -pi / 2.0,
    0.0
]

# The gripper has 9 joints and the positions muss be inside its limits.
# Open configuration: all finger joints relaxed/open.
OPEN_JOINT_STATE = [
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0
]

# Closed configuration: moderate closure value to grasp the red prism.
# These values are conservative to avoid exceeding typical joint limits.
CLOSE_JOINT_STATE = [
    0.75, 0.75, 0.75,
    0.75, 0.75, 0.75,
    0.75, 0.75, 0.75
]


# -------------------------------------------------------------------------
# BLOCK 2: Cartesian grasping parameters.
# These values define the approach height, grasp height, and lift height used
# by the grasping state machine.
# -------------------------------------------------------------------------

APPROACH_HEIGHT = 0.45
GRASP_HEIGHT = 0.36
LIFT_HEIGHT = 0.68

# Partial gripper closing ratio.
# 0.0 = fully open named target.
# 1.0 = fully closed named target.
# A value around 0.45-0.60 is safer for the red prism than forcing "closed".
PARTIAL_GRIPPER_CLOSE_RATIO = 0.55

# Offset between the detected object center and the end-effector target.
# This is needed because the MoveIt end-effector frame is not necessarily
# the exact center between the gripper fingers.
GRASP_TARGET_X_OFFSET = 0.005
GRASP_TARGET_Y_OFFSET = -0.003

OBJECT_SCENE_HEIGHT = 0.10
OBJECT_BOX_SIZE = (0.10, 0.10, 0.20)

MAX_GRASP_PLANNING_ATTEMPTS = 3



USE_HOME_BEFORE_GRASP = False
USE_CARTESIAN_PATH_FOR_GRASP = True

CARTESIAN_EEF_STEP = 0.01
MIN_CARTESIAN_PATH_FRACTION = 0.70



class MoveUR5Node(object):

    def __init__(self):
        super(MoveUR5Node, self).__init__()

        ## Initialization of `moveit_commander`_ and the ROS node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_UR5_node", anonymous=True)

        # -------------------------------------------------------------------------
        # BLOCK 3: Runtime parameters.
        # These parameters allow the UR5 global reference to be adjusted without
        # editing the Python file.
        # -------------------------------------------------------------------------

        ROBOT_POSITION.x = rospy.get_param("~robot_global_x", ROBOT_POSITION.x)
        ROBOT_POSITION.y = rospy.get_param("~robot_global_y", ROBOT_POSITION.y)
        ROBOT_POSITION.z = rospy.get_param("~robot_global_z", ROBOT_POSITION.z)

        self.grasp_target_x_offset = rospy.get_param("~grasp_target_x_offset", GRASP_TARGET_X_OFFSET)
        self.grasp_target_y_offset = rospy.get_param("~grasp_target_y_offset", GRASP_TARGET_Y_OFFSET)

        self.partial_gripper_close_ratio = rospy.get_param(
            "~partial_gripper_close_ratio",
            PARTIAL_GRIPPER_CLOSE_RATIO
        )
        
        # -------------------------------------------------------------------------
        # BLOCK: Runtime movement strategy parameters.
        # These parameters allow switching between direct Cartesian movement and
        # standard MoveIt pose planning without editing the file again.
        # -------------------------------------------------------------------------

        self.use_home_before_grasp = rospy.get_param("~use_home_before_grasp", USE_HOME_BEFORE_GRASP)
        self.use_cartesian_path_for_grasp = rospy.get_param("~use_cartesian_path_for_grasp", USE_CARTESIAN_PATH_FOR_GRASP)


        rospy.loginfo(
            "Using UR5 global position x=%.3f, y=%.3f, z=%.3f",
            ROBOT_POSITION.x,
            ROBOT_POSITION.y,
            ROBOT_POSITION.z
        )
        
        rospy.loginfo(
            "Using grasp target offset x=%.3f, y=%.3f",
            self.grasp_target_x_offset,
            self.grasp_target_y_offset
        )

        rospy.loginfo(
            "Using partial gripper close ratio=%.3f",
            self.partial_gripper_close_ratio
        )

        rospy.loginfo(
            "Using home-before-grasp=%s, cartesian-path-for-grasp=%s",
            str(self.use_home_before_grasp),
            str(self.use_cartesian_path_for_grasp)
        )

        ## Initialize "RobotCommander". It gives information about the robot, such as its cinematic and the joint state.
        self.robot = moveit_commander.RobotCommander()

        ## Initizalize "PlanningSceneInterface". It offers a remote interface to get and set the scene where the robot works.
        self.scene = moveit_commander.PlanningSceneInterface(synchronous=True)

        ## Initialize "MoveGroupCommander". It is an interface for planning the position of a joint group.
        ## The robot UR5 has three groups:
        ## "arm": The joints of the robotic arm.
        ## "gripper": The joints of the gripper to open and close it.
        ## "gripper_mode": The gripper has differents modes, that allosw to move the fingers in other directions.
        ## This last group doesn't need to  be used in this assignment.
        self.move_arm = moveit_commander.MoveGroupCommander("arm", wait_for_servers=30.0)
        self.move_gripper = moveit_commander.MoveGroupCommander("gripper", wait_for_servers=30.0)

        # -------------------------------------------------------------------------
        # BLOCK 4: MoveIt planning configuration.
        # The planning time, number of attempts, and motion scaling factors are
        # configured to reduce random planning failures and make motion safer.
        # -------------------------------------------------------------------------

        self.move_arm.set_planning_time(10.0)
        self.move_arm.set_num_planning_attempts(10)
        self.move_arm.set_max_velocity_scaling_factor(0.25)
        self.move_arm.set_max_acceleration_scaling_factor(0.25)

        self.move_gripper.set_planning_time(5.0)
        self.move_gripper.set_num_planning_attempts(5)

        # Variables
        self.box_name = ""
        self.object_position = geometry_msgs.msg.Point()
        self.object_detected = False
        self.task_state = "WAIT_FOR_OBJECT"
        self.grasp_orientation = None

        # -------------------------------------------------------------------------
        # BLOCK 5: Scene reset and initial robot configuration.
        # The gripper is opened first, and the arm is moved to a known home state.
        # -------------------------------------------------------------------------

        #Remove all the objects in the scene, if there are.
        self.scene.remove_attached_object(self.move_arm.get_end_effector_link())
        self.scene.remove_world_object()
        rospy.sleep(1.0)

        #Go to home position
        # The gripper state is obtained from MoveIt at runtime to avoid invalid
        # hardcoded joint targets.
        self.runtime_open_gripper_state = self.move_gripper.get_current_joint_values()
        self.runtime_close_gripper_state = self.build_runtime_gripper_close_state()

        rospy.loginfo("Runtime open gripper state: {}".format(self.runtime_open_gripper_state))
        rospy.loginfo("Runtime close gripper state: {}".format(self.runtime_close_gripper_state))

        self.go_to_named_gripper_state(["open", "Open", "opened", "Opened"], "open gripper")

        if self.use_home_before_grasp:
            rospy.loginfo("Moving arm to HOME before grasp.")
            self.go_to_joint_arm_state(HOME_JOINT_STATE)
        else:
            rospy.loginfo("Skipping HOME movement because the object is already close to the gripper.")

        # Save the current end-effector orientation after reaching HOME.
        # This orientation is reused during approach, descent, and lifting.
        self.grasp_orientation = self.move_arm.get_current_pose().pose.orientation

        #TODO Consider additional subscribers and publishers for the communication between both robots.

        ## Publishers definition

        # Subscribers definition
        self.position_subscriber = rospy.Subscriber("/pose_array",
            geometry_msgs.msg.PoseArray, self.detection_callback, queue_size=1)

    def detection_callback(self, pose_array):
        # Freeze the target once the grasping sequence has started.
        # This prevents the RGB-D callback from changing the target while the arm is moving.
        if hasattr(self, "task_state") and self.task_state != "WAIT_FOR_OBJECT":
            return

        # Check if the object has been detected. It is communicated through the position in z.

        # -------------------------------------------------------------------------
        # BLOCK 6: Object detection callback.
        # The red object is expected in poses[0]. If z < 0, the detection is invalid.
        # -------------------------------------------------------------------------

        if len(pose_array.poses) == 0:
            return

        object_global_position = pose_array.poses[0].position

        if object_global_position.z < 0:
            self.object_detected = False
            return

        self.object_position = self.convert_global_object_to_arm_frame(object_global_position)
        self.object_detected = True

    def convert_global_object_to_arm_frame(self, object_global_position):
        # -------------------------------------------------------------------------
        # BLOCK 7: Global-to-UR5 coordinate conversion.
        # The RGB-D localization is converted into the local reference used by the
        # UR5 planning task.
        # -------------------------------------------------------------------------

        object_arm_position = geometry_msgs.msg.Point()
        object_arm_position.x = object_global_position.x - ROBOT_POSITION.x
        object_arm_position.y = object_global_position.y - ROBOT_POSITION.y
        object_arm_position.z = OBJECT_SCENE_HEIGHT

        return object_arm_position

    def get_grasp_target_position(self):
        # -------------------------------------------------------------------------
        # BLOCK: Grasp target correction.
        # The RGB-D system detects the center of the red object, but the Cartesian
        # target sent to MoveIt corresponds to the end-effector frame. Since the
        # end-effector frame is not exactly the center between the fingers, a small
        # x/y offset is applied to align the gripper around the object.
        # -------------------------------------------------------------------------

        grasp_target_position = geometry_msgs.msg.Point()
        grasp_target_position.x = self.object_position.x + self.grasp_target_x_offset
        grasp_target_position.y = self.object_position.y + self.grasp_target_y_offset
        grasp_target_position.z = self.object_position.z

        return grasp_target_position

    def print_grasp_alignment_error(self, grasp_pose):
        # -------------------------------------------------------------------------
        # BLOCK: Grasp alignment diagnostic after descent.
        # This function compares the current end-effector position with the detected
        # red object position and the commanded grasp target. It is used to estimate
        # how much x/y offset should be applied in the next execution.
        # -------------------------------------------------------------------------

        current_pose = self.move_arm.get_current_pose().pose
        current_position = current_pose.position

        object_position = self.object_position
        target_position = grasp_pose.position

        target_error_x = current_position.x - target_position.x
        target_error_y = current_position.y - target_position.y
        target_error_z = current_position.z - target_position.z
        target_error_xy = (target_error_x ** 2 + target_error_y ** 2) ** 0.5
        target_error_3d = (target_error_x ** 2 + target_error_y ** 2 + target_error_z ** 2) ** 0.5

        object_error_x = current_position.x - object_position.x
        object_error_y = current_position.y - object_position.y
        object_error_z = current_position.z - object_position.z
        object_error_xy = (object_error_x ** 2 + object_error_y ** 2) ** 0.5
        object_error_3d = (object_error_x ** 2 + object_error_y ** 2 + object_error_z ** 2) ** 0.5

        suggested_x_offset = self.grasp_target_x_offset - object_error_x
        suggested_y_offset = self.grasp_target_y_offset - object_error_y

        rospy.logwarn("========== GRASP ALIGNMENT DIAGNOSTIC ==========")
        rospy.logwarn("Planning frame: %s", self.move_arm.get_planning_frame())

        rospy.logwarn(
            "Detected object position:     x=%.3f, y=%.3f, z=%.3f",
            object_position.x,
            object_position.y,
            object_position.z
        )

        rospy.logwarn(
            "Commanded grasp target:        x=%.3f, y=%.3f, z=%.3f",
            target_position.x,
            target_position.y,
            target_position.z
        )

        rospy.logwarn(
            "Current end-effector position: x=%.3f, y=%.3f, z=%.3f",
            current_position.x,
            current_position.y,
            current_position.z
        )

        rospy.logwarn(
            "Error to commanded target: dx=%.3f, dy=%.3f, dz=%.3f | xy=%.3f m | 3d=%.3f m",
            target_error_x,
            target_error_y,
            target_error_z,
            target_error_xy,
            target_error_3d
        )

        rospy.logwarn(
            "Error to detected object:  dx=%.3f, dy=%.3f, dz=%.3f | xy=%.3f m | 3d=%.3f m",
            object_error_x,
            object_error_y,
            object_error_z,
            object_error_xy,
            object_error_3d
        )

        rospy.logwarn(
            "Current offsets:   x=%.3f, y=%.3f",
            self.grasp_target_x_offset,
            self.grasp_target_y_offset
        )

        rospy.logwarn(
            "Suggested offsets: x=%.3f, y=%.3f",
            suggested_x_offset,
            suggested_y_offset
        )

        rospy.logwarn(
            "Suggested command: rosrun robotica_inteligente move_robot_arm.py _grasp_target_x_offset:=%.3f _grasp_target_y_offset:=%.3f",
            suggested_x_offset,
            suggested_y_offset
        )

        rospy.logwarn("================================================")

    def is_grasp_pose_acceptable(self, grasp_pose):
        # -------------------------------------------------------------------------
        # BLOCK: Verified grasp-pose acceptance.
        # MoveIt/Gazebo can report CONTROL_FAILED even when the end-effector has
        # physically reached a valid grasping pose. This function checks the real
        # current pose before allowing the gripper to close.
        # -------------------------------------------------------------------------

        current_pose = self.move_arm.get_current_pose().pose
        current_position = current_pose.position

        object_position = self.object_position
        target_position = grasp_pose.position

        # Distance to the commanded grasp target.
        target_error_x = current_position.x - target_position.x
        target_error_y = current_position.y - target_position.y
        target_error_z = current_position.z - target_position.z
        target_error_xy = (target_error_x ** 2 + target_error_y ** 2) ** 0.5
        target_error_3d = (target_error_x ** 2 + target_error_y ** 2 + target_error_z ** 2) ** 0.5

        # XY distance to the detected red object.
        object_error_x = current_position.x - object_position.x
        object_error_y = current_position.y - object_position.y
        object_error_xy = (object_error_x ** 2 + object_error_y ** 2) ** 0.5

        # Tolerances in metres.
        max_target_3d_error = 0.025
        max_object_xy_error = 0.025
        max_z_error = 0.030

        acceptable = (
            target_error_3d <= max_target_3d_error and
            object_error_xy <= max_object_xy_error and
            abs(target_error_z) <= max_z_error
        )

        rospy.logwarn(
            "Verified grasp check: target_3d_error=%.3f m, object_xy_error=%.3f m, z_error=%.3f m, acceptable=%s",
            target_error_3d,
            object_error_xy,
            abs(target_error_z),
            acceptable
        )

        return acceptable

    def is_lift_pose_acceptable(self, lift_pose):
        # -------------------------------------------------------------------------
        # BLOCK: Verified lift acceptance.
        # Gazebo/MoveIt may report CONTROL_FAILED even when the object has been
        # physically lifted. This function checks the real current end-effector
        # pose and accepts the lift if the gripper is high enough.
        # -------------------------------------------------------------------------

        current_pose = self.move_arm.get_current_pose().pose
        current_position = current_pose.position
        target_position = lift_pose.position

        z_error = abs(current_position.z - target_position.z)
        xy_error = (
            (current_position.x - target_position.x) ** 2 +
            (current_position.y - target_position.y) ** 2
        ) ** 0.5

        # Conservative acceptance thresholds.
        min_acceptable_lift_z = 0.60
        max_xy_error = 0.10
        max_z_error = 0.10

        acceptable = (
            current_position.z >= min_acceptable_lift_z and
            xy_error <= max_xy_error and
            z_error <= max_z_error
        )

        rospy.logwarn(
            "Verified lift check: current_z=%.3f m, target_z=%.3f m, xy_error=%.3f m, z_error=%.3f m, acceptable=%s",
            current_position.z,
            target_position.z,
            xy_error,
            z_error,
            acceptable
        )

        return acceptable

    def go_to_joint_arm_state(self, joint_goal):
        ## Movement to a joint position of the arm.
        ## The order of the joints is the following: shoulder_pan_joint, shoulder_lift_join,
        ## elbow_joint, wrist1_joint, wrist2_joint, wrist3:joint
        print("Target joint position of the robotic arm:")
        print(joint_goal)

        if len(joint_goal) != 6:
            print("Joint position not valid, UR5 has 6 joints.")
            return False

        # The command go() can be used with joint values, poses or without parameters,
        # if the target has been defined for the group.
        self.move_arm.go(joint_goal, wait=True)

        # The stop() function ensures that there isn't residual movement.
        self.move_arm.stop()

        # Check if the robot has reached the target.
        current_joints = self.move_arm.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.015)

    def go_to_joint_gripper_state(self, joint_goal):
        ## Movement to a joint position of the gripper.
        # The gripper has several joints. The exact number is obtained from MoveIt
        # to avoid assuming an invalid hardcoded size.
        print("Target joint position of the gripper:")
        print(joint_goal)

        current_joint_values = self.move_gripper.get_current_joint_values()

        if len(joint_goal) != len(current_joint_values):
            print("Joint position not valid for this MoveIt gripper group.")
            print("Expected {} joints, received {} joints.".format(len(current_joint_values), len(joint_goal)))
            return False

        try:
            self.move_gripper.go(joint_goal, wait=True)
            self.move_gripper.stop()
            return True
        except Exception as error:
            print("Gripper movement failed:")
            print(error)
            return False

    def build_runtime_gripper_close_state(self):
        # -------------------------------------------------------------------------
        # BLOCK: Runtime gripper close state.
        # The current state is used as the base. A small conservative offset is used
        # so that the gripper attempts to close without relying on invalid limits.
        # -------------------------------------------------------------------------

        current_joint_values = self.move_gripper.get_current_joint_values()
        close_joint_values = []

        for joint_value in current_joint_values:
            close_joint_values.append(joint_value + 0.15)

        return close_joint_values

    def go_to_named_gripper_state(self, candidate_target_names, description):
        # -------------------------------------------------------------------------
        # BLOCK: Named gripper target execution.
        # Some MoveIt configurations define valid named states such as "open" and
        # "close". This function uses them when available.
        # -------------------------------------------------------------------------

        available_named_targets = self.move_gripper.get_named_targets()
        rospy.loginfo("Available gripper named targets: {}".format(available_named_targets))

        for target_name in candidate_target_names:
            if target_name in available_named_targets:
                rospy.loginfo("Executing named gripper target '{}' for {}".format(target_name, description))
                try:
                    self.move_gripper.set_named_target(target_name)
                    success = self.move_gripper.go(wait=True)
                    self.move_gripper.stop()
                    return success
                except Exception as error:
                    rospy.logwarn("Named gripper target '{}' failed: {}".format(target_name, error))
                    return False

        rospy.logwarn("No named target found for {}. Candidates: {}".format(description, candidate_target_names))
        return False


    def go_to_partial_gripper_close_state(self):
        # -------------------------------------------------------------------------
        # BLOCK: Partial gripper closing.
        # The named target "closed" may close too aggressively and destabilize the
        # red object. This function interpolates between the named "open" and
        # "closed" targets to obtain a softer grasp.
        # -------------------------------------------------------------------------

        available_named_targets = self.move_gripper.get_named_targets()

        if "open" not in available_named_targets or "closed" not in available_named_targets:
            rospy.logwarn("Named gripper targets 'open' and 'closed' are not both available.")
            return self.go_to_safe_gripper_state(self.runtime_close_gripper_state, "partial close gripper fallback")

        active_joint_names = self.move_gripper.get_active_joints()

        open_target_dict = self.move_gripper.get_named_target_values("open")
        closed_target_dict = self.move_gripper.get_named_target_values("closed")

        partial_close_joint_goal = []

        for joint_name in active_joint_names:
            open_value = open_target_dict[joint_name]
            closed_value = closed_target_dict[joint_name]

            partial_value = open_value + self.partial_gripper_close_ratio * (closed_value - open_value)
            partial_close_joint_goal.append(partial_value)

        rospy.loginfo("Partial gripper close target:")
        rospy.loginfo(partial_close_joint_goal)

        return self.go_to_safe_gripper_state(partial_close_joint_goal, "partial close gripper")

    def go_to_safe_gripper_state(self, joint_goal, description):
        # -------------------------------------------------------------------------
        # BLOCK: Safe gripper movement wrapper.
        # This wrapper prevents the whole node from crashing if the numeric gripper
        # target is rejected by MoveIt.
        # -------------------------------------------------------------------------

        rospy.loginfo("Executing gripper command: {}".format(description))

        try:
            return self.go_to_joint_gripper_state(joint_goal)
        except Exception as error:
            rospy.logwarn("Gripper command failed: {}".format(description))
            rospy.logwarn(str(error))
            return False

    def go_to_pose_arm_goal(self, pose_goal):
        ## Movement to a cartesian pose, defined by the target position and orientation of the UR5 wrist.
        print("Target cartesian pose:")
        print(pose_goal)

        # We set the target pose and we use the command go() to plan and execute
        # the movement. The function returns if it has perform it.
        self.move_arm.set_start_state_to_current_state()
        self.move_arm.set_pose_target(pose_goal)
        success = self.move_arm.go(wait=True)

       # The stop() function ensures that there isn't residual movement.
        self.move_arm.stop()

        # It is good to always clean the target pose.
        self.move_arm.clear_pose_targets()

        # Check if the robot has reached the target.
        current_pose = self.move_arm.get_current_pose().pose
        return success and all_close(pose_goal, current_pose, 0.1)

    def go_to_pose_cartesian_path(self, pose_goal, description):
        # -------------------------------------------------------------------------
        # BLOCK: Direct Cartesian path execution.
        # This moves the end-effector through a straight Cartesian interpolation
        # from the current pose to the target pose. It avoids unnecessary large
        # rotations caused by unconstrained IK pose planning.
        # -------------------------------------------------------------------------

        rospy.loginfo("Computing Cartesian path for: %s", description)

        self.move_arm.set_start_state_to_current_state()

        waypoints = []
        waypoints.append(copy.deepcopy(pose_goal))

        plan, fraction = self.move_arm.compute_cartesian_path(
            waypoints,
            0.01,
            True
        )

        rospy.loginfo(
            "Cartesian path fraction for %s: %.3f",
            description,
            fraction
        )

        if fraction < MIN_CARTESIAN_PATH_FRACTION:
            rospy.logwarn(
                "Cartesian path rejected for %s because fraction %.3f is below %.3f",
                description,
                fraction,
                MIN_CARTESIAN_PATH_FRACTION
            )
            return False

        success = self.move_arm.execute(plan, wait=True)

        self.move_arm.stop()
        self.move_arm.clear_pose_targets()

        current_pose = self.move_arm.get_current_pose().pose

        return success and all_close(pose_goal, current_pose, 0.08)

    def go_to_pose_with_retries(self, pose_goal, description):
        # -------------------------------------------------------------------------
        # BLOCK 8: Robust Cartesian motion execution.
        # MoveIt can occasionally fail to plan even for reachable poses, so the same
        # target is attempted several times before declaring failure.
        # -------------------------------------------------------------------------

        for attempt_number in range(1, MAX_GRASP_PLANNING_ATTEMPTS + 1):
            rospy.loginfo(
                "Planning %s. Attempt %d/%d",
                description,
                attempt_number,
                MAX_GRASP_PLANNING_ATTEMPTS
            )

            if self.use_cartesian_path_for_grasp:
                motion_success = self.go_to_pose_cartesian_path(pose_goal, description)
            else:
                motion_success = self.go_to_pose_arm_goal(pose_goal)

            if motion_success:
                rospy.loginfo("Motion succeeded: %s", description)
                return True

            rospy.logwarn("Motion failed: %s", description)
            rospy.sleep(0.5)

        return False

    def build_grasp_pose(self, x_position, y_position, z_position):
        # -------------------------------------------------------------------------
        # BLOCK 9: Cartesian pose construction.
        # The target position is updated while preserving the saved end-effector
        # orientation from the home configuration.
        # -------------------------------------------------------------------------

        target_pose = geometry_msgs.msg.Pose()
        target_pose.position.x = x_position
        target_pose.position.y = y_position
        target_pose.position.z = z_position

        if self.grasp_orientation is not None:
            target_pose.orientation = self.grasp_orientation
        else:
            target_pose.orientation = self.move_arm.get_current_pose().pose.orientation

        return target_pose

    def add_object(self, object_position):
        ## Add the object to the scene. The object position (x,y)
        ## should be respect the robot's base.
        print("Adding the object at the position x={}, y={}".format(object_position.x, object_position.y))
        object_pose = geometry_msgs.msg.PoseStamped()
        object_pose.header.frame_id = "world"
        object_pose.pose.position = copy.deepcopy(object_position)
        object_pose.pose.orientation.w = 1.0
        object_pose.pose.position.z = OBJECT_SCENE_HEIGHT
        self.box_name = "object"
        self.scene.add_box(self.box_name, object_pose, size=OBJECT_BOX_SIZE)
        rospy.sleep(0.5)

    def attach_object(self):
        ## Attach the object to the gripper. Manipulate objects requires that the robot be able to touch them
        ## while the planner doesn't consider the contact as a collision.
        touch_links = self.robot.get_link_names(group="gripper")
        self.scene.attach_box(self.move_arm.get_end_effector_link(), self.box_name, touch_links=touch_links)
        rospy.sleep(0.5)

    def detach_object(self):
        ## Detach the object.
        self.scene.remove_attached_object(self.move_arm.get_end_effector_link(), name=self.box_name)
        rospy.sleep(0.5)

    def remove_object(self):
        ## Remove the object, it should be detached previously.
        self.scene.remove_world_object(self.box_name)
        rospy.sleep(0.5)

    # TODO Define the functions to perform the task of grap the object and leave it in the mobile robot.
    # Consider the use of a state machine to coordinate the process.

    def wait_for_planning_scene_update(self, box_is_known=False, box_is_attached=False, timeout=4.0):
        start_time = rospy.get_time()

        while (rospy.get_time() - start_time < timeout) and not rospy.is_shutdown():
            attached_objects = self.scene.get_attached_objects([self.box_name])
            is_attached = len(attached_objects.keys()) > 0

            known_objects = self.scene.get_known_object_names()
            is_known = self.box_name in known_objects

            if is_attached == box_is_attached and is_known == box_is_known:
                return True

            rospy.sleep(0.1)

        return False


    def add_and_attach_object_to_moveit_scene(self):
        # -------------------------------------------------------------------------
        # Real MoveIt attach.
        # This adds the red prism as a collision object and then attaches it to the
        # end-effector, so RViz shows it as an attached object.
        # -------------------------------------------------------------------------

        self.box_name = "object"

        end_effector_link = self.move_arm.get_end_effector_link()
        planning_frame = self.move_arm.get_planning_frame()

        # Remove previous copies if they exist.
        self.scene.remove_attached_object(end_effector_link, name=self.box_name)
        self.scene.remove_world_object(self.box_name)
        rospy.sleep(0.5)

        object_pose = geometry_msgs.msg.PoseStamped()
        object_pose.header.frame_id = planning_frame
        object_pose.header.stamp = rospy.Time.now()

        object_pose.pose.position.x = self.object_position.x
        object_pose.pose.position.y = self.object_position.y
        object_pose.pose.position.z = OBJECT_SCENE_HEIGHT
        object_pose.pose.orientation.w = 1.0

        self.scene.add_box(
            self.box_name,
            object_pose,
            size=OBJECT_BOX_SIZE
        )

        box_added = self.wait_for_planning_scene_update(
            box_is_known=True,
            box_is_attached=False,
            timeout=4.0
        )

        if not box_added:
            rospy.logwarn("The object was not confirmed as a known object in MoveIt.")
            return False

        # Important:
        # Use all robot links as touch links. This prevents MoveIt from considering
        # the grasped object as colliding with the gripper/fingers/wrist after closing.
        touch_links = self.robot.get_link_names()
        touch_links.append(end_effector_link)
        touch_links = list(set(touch_links))

        self.scene.attach_box(
            end_effector_link,
            self.box_name,
            touch_links=touch_links
        )

        box_attached = self.wait_for_planning_scene_update(
            box_is_known=False,
            box_is_attached=True,
            timeout=4.0
        )

        return box_attached

    def execute_grasp_state_machine(self):
        # -------------------------------------------------------------------------
        # BLOCK 10: Part 1 grasping state machine.
        # The robot waits for the detected object, opens the gripper, approaches
        # the object, descends, closes the gripper, attaches the object, and lifts it.
        # -------------------------------------------------------------------------

        if self.task_state == "WAIT_FOR_OBJECT":
            if not self.object_detected:
                rospy.loginfo_throttle(2.0, "Waiting for a valid red object detection on /pose_array...")
                return

            rospy.loginfo(
                "Detected object in UR5 frame: x=%.3f, y=%.3f, z=%.3f",
                self.object_position.x,
                self.object_position.y,
                self.object_position.z
            )

            self.task_state = "ADD_OBJECT_TO_SCENE"
            return

        if self.task_state == "ADD_OBJECT_TO_SCENE":
            # Do not add the red cube as a collision object before the descent.
            # If the object is already in the MoveIt planning scene, the planner
            # avoids it and may refuse to descend close enough for grasping.
            rospy.loginfo("Skipping pre-grasp collision object insertion.")
            self.task_state = "OPEN_GRIPPER"
            return

        if self.task_state == "OPEN_GRIPPER":
            if not self.go_to_named_gripper_state(["open", "Open", "opened", "Opened"], "open gripper"):
                self.go_to_safe_gripper_state(self.runtime_open_gripper_state, "open gripper")
            self.task_state = "APPROACH_OBJECT"
            return

        if self.task_state == "APPROACH_OBJECT":
            grasp_target_position = self.get_grasp_target_position()

            rospy.loginfo(
                "Approach target after offset: x=%.3f, y=%.3f, z=%.3f",
                grasp_target_position.x,
                grasp_target_position.y,
                APPROACH_HEIGHT
            )

            approach_pose = self.build_grasp_pose(
                grasp_target_position.x,
                grasp_target_position.y,
                APPROACH_HEIGHT
            )

            if self.go_to_pose_with_retries(approach_pose, "approach object from above"):
                self.task_state = "DESCEND_TO_OBJECT"
            else:
                rospy.logwarn("Could not reach approach pose. Returning to WAIT_FOR_OBJECT.")
                self.task_state = "WAIT_FOR_OBJECT"

            return

        if self.task_state == "DESCEND_TO_OBJECT":
            grasp_target_position = self.get_grasp_target_position()

            rospy.loginfo(
                "Grasp target after offset: x=%.3f, y=%.3f, z=%.3f",
                grasp_target_position.x,
                grasp_target_position.y,
                GRASP_HEIGHT
            )

            grasp_pose = self.build_grasp_pose(
                grasp_target_position.x,
                grasp_target_position.y,
                GRASP_HEIGHT
            )

            descent_success = self.go_to_pose_with_retries(grasp_pose, "descend to grasp height")

            # Print diagnostic information even if MoveIt reports CONTROL_FAILED.
            self.print_grasp_alignment_error(grasp_pose)

            # Do not close blindly. Close only if MoveIt succeeded OR if the real
            # end-effector pose is verified to be close enough to the grasp target.
            descent_acceptable = self.is_grasp_pose_acceptable(grasp_pose)

            if descent_success or descent_acceptable:
                if not descent_success:
                    rospy.logwarn(
                        "MoveIt reported failure, but the measured end-effector pose is acceptable. Proceeding to close gripper."
                    )
                self.task_state = "CLOSE_GRIPPER"
            else:
                rospy.logwarn("Could not reach a verified grasp pose. Returning to APPROACH_OBJECT.")
                self.task_state = "APPROACH_OBJECT"

            return

        if self.task_state == "CLOSE_GRIPPER":
            close_success = self.go_to_partial_gripper_close_state()

            if close_success:
                self.task_state = "ATTACH_OBJECT"
            else:
                rospy.logwarn("Partial gripper close failed. Returning to APPROACH_OBJECT.")
                self.task_state = "APPROACH_OBJECT"

            return

        if self.task_state == "ATTACH_OBJECT":
            rospy.loginfo("Adding and attaching object in MoveIt planning scene.")

            attach_success = self.add_and_attach_object_to_moveit_scene()

            if attach_success:
                rospy.loginfo("Object attached in MoveIt planning scene.")
                self.task_state = "LIFT_OBJECT"
            else:
                rospy.logwarn("MoveIt object attach was not confirmed. Retrying attach before lifting.")
                self.task_state = "ATTACH_OBJECT"
                rospy.sleep(0.5)

            return

        if self.task_state == "LIFT_OBJECT":
            grasp_target_position = self.get_grasp_target_position()

            lift_pose = self.build_grasp_pose(
                grasp_target_position.x,
                grasp_target_position.y,
                LIFT_HEIGHT
            )

            lift_success = self.go_to_pose_with_retries(lift_pose, "lift grasped object")

            if lift_success or self.is_lift_pose_acceptable(lift_pose):
                if not lift_success:
                    rospy.logwarn(
                        "MoveIt reported lift failure, but the measured end-effector height is acceptable. Marking Part 1 as completed."
                    )

                self.task_state = "FINISHED"
            else:
                rospy.logwarn("Could not verify that the object was lifted. Keeping current state for safety.")

            return

        if self.task_state == "FINISHED":
            rospy.loginfo_throttle(5.0, "Part 1 completed: object grasped and lifted.")
            return

    def run(self):
        #Control loop
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            #TODO call the functions to perform the task.
            self.execute_grasp_state_machine()
            rate.sleep()


def all_close(goal, actual, tolerance):
    """
    A method to check if the values of both lists are inside a tolerance threshold.
    For a pose, the angle between both quaternions is also compared.
    @param: goal       A list of floats, a Pose or a PoseStamped or Point
    @param: actual     A list of floats, a Pose or a PoseStamped or Point
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)

        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))

        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)

        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    elif type(goal) is geometry_msgs.msg.Point:
        x0, y0, z0 = actual.x, actual.y, actual.z
        x1, y1, z1 = goal.x, goal.y, goal.z

        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))

        return d <= tolerance

    return True


def main():
    try:
        print("Init move_UR5_node")
        node = MoveUR5Node()
        node.run()

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
