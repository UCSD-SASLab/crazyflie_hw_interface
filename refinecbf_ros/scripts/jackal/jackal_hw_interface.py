#!/usr/bin/env python3

import rospy
import numpy as np

from geometry_msgs.msg import Twist, PoseStamped
from refinecbf_ros.msg import Array
from std_srvs.srv import Empty, EmptyResponse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from template.hw_interface import BaseInterface
import math


class JackalInterface(BaseInterface):
    """
    This class converts the state and control messages from the SafetyFilterNode to the correct type
    for the Jackal.
    Each HW platform should have its own Interface node
    """

    control_out_msg_type = Twist
    external_control_msg_type = Twist

    def __init__(self):
        # Initialize external control parameters
        self.hardware = rospy.get_param("~hardware")
        if self.hardware:
            self.state_msg_type = PoseStamped
        else:
            self.state_msg_type = Twist

        self.external_control = None
        self.external_control_time_buffer = rospy.get_param("/ctr/external_control_buffer", 1.0)  # seconds
        self.external_control_change_time_buffer = rospy.get_param(
            "/ctr/external_control_change_buffer", 5.0
        )  # seconds

        s = rospy.Service('toggle_move', Empty, self.toggle_movement)
        self.is_moving = False
        super().__init__()
        

    def callback_state(self, state_in_msg):
        if self.hardware:
            w = state_in_msg.pose.orientation.w
            x = state_in_msg.pose.orientation.x
            y = state_in_msg.pose.orientation.y
            z = state_in_msg.pose.orientation.z
            xx = state_in_msg.pose.position.x
            yy = state_in_msg.pose.position.y
            # Convert Quaternion to Yaw
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (np.power(y, 2) + np.power(z, 2))) + np.pi / 2 # FIXME: why is this necessary? I think it has something to do with the odom and rviz coordinate frames
            yaw = np.arctan2(np.sin(yaw),np.cos(yaw)) # Remap yaw to -pi to pi range
        else:
            xx = state_in_msg.linear.x
            yy = state_in_msg.linear.y
            yaw = state_in_msg.angular.z + np.pi / 2
            yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
        
        state_out_msg = Array()
        state_out_msg.value = [xx, yy, yaw]
        self.state_pub.publish(state_out_msg)

    def process_safe_control(self, control_in_msg):
        control_in = control_in_msg.value
        control_out_msg = self.control_out_msg_type()
        control_out_msg.linear.x = control_in[1]
        control_out_msg.linear.y = 0.0
        control_out_msg.linear.z = 0.0

        control_out_msg.angular.x = 0.0
        control_out_msg.angular.y = 0.0
        control_out_msg.angular.z = control_in[0]
        return control_out_msg
    
    def toggle_movement(self, req):
        if self.is_moving:
            control_out_msg = self.control_out_msg_type()
            control_out_msg.linear.x = 0.0
            control_out_msg.linear.y = 0.0
            control_out_msg.linear.z = 0.0

            control_out_msg.angular.x = 0.0
            control_out_msg.angular.y = 0.0
            control_out_msg.angular.z = 0.0
            self.safe_control_pub.publish(control_out_msg)

        self.is_moving = not self.is_moving
        rospy.loginfo("Robot moving? {}".format(self.is_moving))
        return EmptyResponse()


    def process_external_control(self, control_in_msg):
        # When nominal control comes through the HW interface, it is a Twist message
        control_out_msg = Array()
        control_out_msg.value = [control_in_msg.angular.z, control_in_msg.linear.x]
        new_val = np.array(control_out_msg.value)
        if (self.external_control is None):# or (not np.allclose(self.external_control, new_val, atol=1e-1, rtol=1e-1)):
            # If the external control has changed, then reset the external control mod timestamp
            self.external_control_mod_ts = rospy.get_time()
            self.external_control = new_val
        
        self.external_control = control_out_msg
        self.external_control_ts = rospy.get_time()
        return control_out_msg
    
    def override_safe_control(self):
        curr_time = rospy.get_time()

        # Determine if external control should be published
        # print(self.external_control)
        # print(self.external_control is not None and (curr_time - self.external_control_ts) <= self.external_control_time_buffer)
        return not self.is_moving 
        #     self.external_control is not None
        #     and (curr_time - self.external_control_ts) <= self.external_control_time_buffer
        # )


if __name__ == "__main__":
    rospy.init_node("interface")
    JackalInterface()
    rospy.spin()
