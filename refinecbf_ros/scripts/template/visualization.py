#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose
from std_msgs.msg import ColorRGBA, Bool
from refinecbf_ros.msg import ValueFunctionMsg, Array, Obstacles
from refinecbf_ros.config import Config
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import warnings

# Suppress the warning
warnings.filterwarnings("ignore", category=UserWarning)


class Visualization:

    def __init__(self):

        # Config:
        config = Config(hj_setup=True)

        self.state_safety_idis = config.safety_states
        self.grid = config.grid

        # Control Dict with Goal Params:
        self.control_dict = rospy.get_param("~/ctr/nominal")

        # Subscriber for SDF and VF:
        self.vf_update_method = rospy.get_param("~vf_update_method")
        sdf_update_topic = rospy.get_param("~topics/sdf_update")
        vf_topic = rospy.get_param("~topics/vf_update")

        if self.vf_update_method == "pubsub":
            self.sdf_update_sub = rospy.Subscriber(
                sdf_update_topic, ValueFunctionMsg, self.callback_sdf_pubsub
            )
            self.vf_update_sub = rospy.Subscriber(vf_topic, ValueFunctionMsg, self.callback_vf_pubsub)
        elif self.vf_update_method == "file":
            self.sdf_update_sub = rospy.Subscriber(sdf_update_topic, Bool, self.callback_sdf_file)
            self.vf_update_sub = rospy.Subscriber(vf_topic, Bool, self.callback_vf_file)
        else:
            raise NotImplementedError("{} is not a valid vf update method".format(self.vf_update_method))
        
        obstacle_update_topic = rospy.get_param("~topics/obstacle_update")
        self.obstacle_update_sub = rospy.Subscriber(obstacle_update_topic,Obstacles,self.callback_obstacle)
        self.active_obstacle_names = []

        # Publisher for Marker messages
        obstacle_marker_topic = rospy.get_param("~topics/obstacle_marker")
        self.obstacle_marker_publisher = rospy.Publisher(obstacle_marker_topic, Marker, queue_size=10)

        # Publisher for SDF
        sdf_marker_topic = rospy.get_param("~topics/sdf_marker")
        self.sdf_marker_publisher = rospy.Publisher(sdf_marker_topic, Marker, queue_size=10)

        # Publisher for VF
        vf_marker_topic = rospy.get_param("~topics/vf_marker")
        self.vf_marker_publisher = rospy.Publisher(vf_marker_topic, Marker, queue_size=10)

        # Publisher for Goal:
        goal_marker_topic = rospy.get_param("~topics/goal_marker")
        self.goal_marker_publisher = rospy.Publisher(goal_marker_topic, Marker, queue_size=10)

        # Subscriber for Robot State:
        cbf_state_topic = rospy.get_param("~topics/cbf_state")
        state_sub = rospy.Subscriber(cbf_state_topic, Array, self.callback_state)

        # load Obstacle and Boundary dictionaries
        self.obstacle_dict = rospy.get_param("~/env/obstacles")
        self.boundary_dict = rospy.get_param("~/env/boundary")

    def clip_state(self, state):
        return np.clip(state, np.array(self.grid.domain.lo) + 0.01, np.array(self.grid.domain.hi) - 0.01)

    def obstacle_marker(self, obstacle, obstacle_marker_id):
        raise NotImplementedError("Must Be Subclassed")

    def sdf_marker(self, points, sdf_marker_id):
        raise NotImplementedError("Must Be Subclassed")

    def vf_marker(self, points, vf_marker_id):
        raise NotImplementedError("Must Be Subclassed")

    def zero_level_set_contour(self, vf):
        raise NotImplementedError("Must Be Subclassed")
    
    def goal_marker(self, control_dict,goal_marker_id):
        raise NotImplementedError("Must Be Subclassed")

    def add_obstacles(self):

        obstacle_marker_id = 1
        if len(self.obstacle_dict) != 0:
            for name, obstacle in self.obstacle_dict.items():
                # Create a Marker message for each obstacle
                marker = self.obstacle_marker(obstacle, obstacle_marker_id,name in self.active_obstacle_names)
                self.obstacle_marker_publisher.publish(marker)
                obstacle_marker_id = obstacle_marker_id + 1

    def update_sdf_contour(self):

        sdf_marker_id = 100
        array_points = self.zero_level_set_contour(self.sdf)

        for i in range(len(array_points)):
            marker = self.sdf_marker(array_points[i], sdf_marker_id + i)
            self.sdf_marker_publisher.publish(marker)

    def update_vf_contour(self):

        vf_marker_id = 200
        array_points = self.zero_level_set_contour(self.vf)

        for i in range(len(array_points)):
            marker = self.vf_marker(array_points[i], vf_marker_id + i)
            self.vf_marker_publisher.publish(marker)

    def add_goal(self):
        goal_marker_id = 300
        marker = self.goal_marker(self.control_dict,goal_marker_id)
        self.goal_marker_publisher.publish(marker)

    def callback_sdf_pubsub(self, sdf_msg):
        self.sdf = np.array(sdf_msg.vf).reshape(self.grid.shape)

    def callback_sdf_file(self, sdf_msg):
        if not sdf_msg.data:
            return
        self.sdf = np.array(np.load("./sdf.npy")).reshape(self.grid.shape)

    def callback_vf_pubsub(self, vf_msg):
        self.vf = np.array(vf_msg.vf).reshape(self.grid.shape)
    
    def callback_vf_file(self, vf_msg):
        if not vf_msg.data:
            return
        self.vf = np.array(np.load("./vf.npy")).reshape(self.grid.shape)

    def callback_obstacle(self, obstacle_msg):
        self.active_obstacle_names = obstacle_msg.obstacle_names

    def callback_state(self, state_msg):
        self.robot_state = jnp.reshape(np.array(state_msg.value)[self.state_safety_idis], (-1, 1)).T
        if hasattr(self, "vf"):
            self.update_vf_contour()
        if hasattr(self,"sdf"):
            self.update_sdf_contour()
        if hasattr(self,"obstacle_dict"):
            self.add_obstacles()
        if hasattr(self,"control_dict"):
            self.add_goal()
