#!/usr/bin/env python

import rospy, tf, rospkg
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import *
import numpy as np

if __name__ == '__main__':
    print("Waiting for gazebo services...")
    rospy.init_node("spawn_random_position")
    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    rospy.wait_for_service("/gazebo/spawn_urdf_model")

    #Spawn object
    spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

    with open(rospkg.RosPack().get_path("robotica_inteligente")+"/models/object/model.sdf", "r") as f:
        object_model = f.read()
        
    # Red object spawn position near the UR5 gripper working area.
    # The object is spawned around a point that is reachable by the UR5 and close
    # to the grasping position previously validated from /pose_array.

    mu = np.array([0.56, -0.25])
    point = mu.copy()

    object_pose   =   Pose(Point(x=point[0], y=point[1],    z=0.1),   Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
    spawn_model("Object", object_model, "object", object_pose, "world")

    #Spawn camera
    spawn_model = rospy.ServiceProxy("/gazebo/spawn_urdf_model", SpawnModel)

    with open(rospkg.RosPack().get_path("robotica_inteligente")+"/models/realsense/model.urdf", "r") as f:
        object_model = f.read()


    object_pose   =   Pose(Point(x=0, y=0,    z=10.0),   Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
    spawn_model("Realsense", object_model, "camera", object_pose, "world")