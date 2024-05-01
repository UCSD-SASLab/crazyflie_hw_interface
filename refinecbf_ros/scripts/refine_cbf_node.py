#!/usr/bin/env python3

import rospy
import numpy as np
import jax.numpy as jnp
from refinecbf_ros.msg import ValueFunctionMsg, Array, HiLoArray
from std_msgs.msg import Bool, Float32
from cbf_opt import ControlAffineASIF
from refine_cbfs import TabularControlAffineCBF
from refinecbf_ros.config import Config
from cbf_opt import ControlAffineASIF, SlackifiedControlAffineASIF


class SafetyFilterNode:
    """
    Docstring missing
    """

    def __init__(self):
        self.initialized_safety_filter = False  # To ensure initialized when callback is triggered
        self.safety_filter_active = rospy.get_param("~safety_filter_active", True)
        vf_topic = rospy.get_param("~topics/vf_update")
        self.vf_update_method = rospy.get_param("~vf_update_method")
        gamma = rospy.get_param("/ctr/cbf/gamma", 1.0)
        slackify_safety_constraint = rospy.get_param("/ctr/cbf/slack", False)

        config = Config(hj_setup=True)
        self.dynamics = config.dynamics
        self.grid = config.grid
        self.safety_states_idis = config.safety_states
        self.safety_controls_idis = config.safety_controls

        if self.vf_update_method == "pubsub":
            self.vf_sub = rospy.Subscriber(vf_topic, ValueFunctionMsg, self.callback_vf_update_pubsub)
        elif self.vf_update_method == "file":
            self.vf_sub = rospy.Subscriber(vf_topic, Bool, self.callback_vf_update_file)
        else:
            raise NotImplementedError("{} is not a valid vf update method".format(self.vf_update_method))
        self.state_topic = rospy.get_param("~topics/state", "/state_array")
        self.state_sub = rospy.Subscriber(self.state_topic, Array, self.callback_state)

        alpha = lambda x: gamma * x
        self.cbf = TabularControlAffineCBF(self.dynamics, grid=self.grid, alpha=alpha)

        if slackify_safety_constraint:
            self.safety_filter_solver = SlackifiedControlAffineASIF(self.dynamics, self.cbf)
        else:  # strict equality
            self.safety_filter_solver = ControlAffineASIF(self.dynamics, self.cbf)

        self.safety_filter_solver.umin = np.array(config.control_space["lo"])
        self.safety_filter_solver.umax = np.array(config.control_space["hi"])
        if not config.disturbance_space["n_dims"] == 0:
            self.safety_filter_solver.dmin = np.array(config.disturbance_space["lo"])
            self.safety_filter_solver.dmax = np.array(config.disturbance_space["hi"])

        nom_control_topic = rospy.get_param("~topics/nominal_control", "/control/nominal")
        self.nominal_control_sub = rospy.Subscriber(nom_control_topic, Array, self.callback_safety_filter)
        self.state = None
        filtered_control_topic = rospy.get_param("~topics/filtered_control", "/control/filtered")
        self.pub_filtered_control = rospy.Publisher(filtered_control_topic, Array, queue_size=1)

        actuation_update_topic = rospy.get_param("~topics/actuation_update", "/env/actuation_update")
        self.actuation_update_sub = rospy.Subscriber(actuation_update_topic, HiLoArray, self.callback_actuation_update)

        if not config.disturbance_space["n_dims"] == 0:
            disturbance_update_topic = rospy.get_param("~topics/disturbance_update")
            self.disturbance_update_sub = rospy.Subscriber(disturbance_update_topic, HiLoArray, self.callback_disturbance_update)

        value_function_topic = rospy.get_param("~topics/value_function", "/visualization/value_function")
        self.value_function_pub = rospy.Publisher(value_function_topic, Float32, queue_size=1)

        if self.safety_filter_active:
            # This has to be done to ensure real-time performance
            self.initialized_safety_filter = False
            self.safety_filter_solver.setup_optimization_problem()
            rospy.loginfo("safety filter is used, but not initialized yet")

        else:
            self.initialized_safety_filter = True
            self.safety_filter_solver = lambda state, nominal_control: nominal_control
            rospy.logwarn("No safety filter, be careful!")

    def callback_actuation_update(self, msg):
        self.safety_filter_solver.umin = np.array(msg.lo)
        self.safety_filter_solver.umax = np.array(msg.hi)
    
    def callback_disturbance_update(self, msg):
        self.safety_filter_solver.dmin = np.array(msg.lo)
        self.safety_filter_solver.dmax = np.array(msg.hi)

    def callback_vf_update_file(self, vf_msg):
        if not vf_msg.data:
            return
        self.cbf.vf_table = np.array(np.load("./vf.npy")).reshape(self.grid.shape)
        if not self.initialized_safety_filter:
            rospy.loginfo("Initialized safety filter")
            self.initialized_safety_filter = True

    def callback_vf_update_pubsub(self, vf_msg):
        self.cbf.vf_table = np.array(vf_msg.vf).reshape(self.grid.shape)
        if not self.initialized_safety_filter:
            rospy.loginfo("Initialized safety filter")
            self.initialized_safety_filter = True

    def callback_safety_filter(self, control_msg):
        nom_control = np.array(control_msg.value)
        if self.state is None:
            rospy.loginfo(" State not set yet, no control published")
            return
        if not self.initialized_safety_filter:
            safety_control_msg = control_msg
            rospy.logwarn_throttle_identical(5.0, "Safety filter not initialized yet, outputting nominal control")
        else:
            nom_control_active = nom_control[self.safety_controls_idis]
            safety_control_msg = Array()
            if hasattr(self.safety_filter_solver, "cbf"):
                vf = np.array(self.safety_filter_solver.cbf.vf(self.state.copy(), 0.0)).item()
                self.value_function_pub.publish(vf)
                # rospy.loginfo_throttle_identical(1.0, "value at current state:{:.2f}".format(vf))
            safety_control_active = self.safety_filter_solver(self.state.copy(), nominal_control=np.array([nom_control_active]))
            safety_control = nom_control.copy()

            safety_control[self.safety_controls_idis] = safety_control_active[0]
            safety_control_msg.value = safety_control.tolist()  # Ensures compatibility

        self.pub_filtered_control.publish(safety_control_msg)

    def callback_state(self, state_est_msg):
        self.state = np.array(state_est_msg.value)[self.safety_states_idis]


if __name__ == "__main__":
    rospy.init_node("safety_filter_node")
    safety_filter = SafetyFilterNode()
    rospy.spin()
