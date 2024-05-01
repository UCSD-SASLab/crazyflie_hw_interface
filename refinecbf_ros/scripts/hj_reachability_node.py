#!/usr/bin/env python3

import rospy
import numpy as np
import hj_reachability as hj
import jax.numpy as jnp
from threading import Lock
from refinecbf_ros.msg import ValueFunctionMsg, HiLoArray
from refinecbf_ros.config import Config
from refinecbf_ros.config import QuadraticCBF
from refine_cbfs import HJControlAffineDynamics
from std_msgs.msg import Bool
from refine_cbfs import (
    HJControlAffineDynamics,
    TabularControlAffineCBF,
    TabularTVControlAffineCBF,
    utils,
)
import os


class HJReachabilityNode:
    """
    HJReachabilityNode is a ROS node that computes the Hamilton-Jacobi reachability for a robot.

    Subscribers:
    - disturbance_update_sub (~topics/disturbance_update): Updates the disturbance.
    - actuation_update_sub (~topics/actuation_update): Updates the actuation.
    - sdf_update_sub (~topics/sdf_update): Updates the obstacles.

    Publishers:
    - vf_pub (~topics/vf_update): Publishes the value function.
    """

    def __init__(self) -> None:
        """
        Initializes the HJReachabilityNode. It sets up ROS subscribers for disturbance, actuation, and obstacle updates,
        and a publisher for the value function. It also initializes the Hamilton-Jacobi dynamics and the value function.
        """
        # Load configuration
        config = Config(hj_setup=True)
        # Initialize dynamics, grid, and Hamilton-Jacobi dynamics
        self.dynamics = config.dynamics
        self.grid = config.grid
        self.hj_dynamics = config.hj_dynamics
        self.control_space = self.hj_dynamics.control_space
        self.disturbance_space = self.hj_dynamics.disturbance_space

        self.vf_update_method = rospy.get_param("~vf_update_method")

        self.vf_update_accuracy = rospy.get_param("~vf_update_accuracy", "medium")
        # Initialize a lock for thread-safe value function updates
        self.vf_lock = Lock()
        # Get initial safe space and setup solver
        sdf_update_topic = rospy.get_param("~topics/sdf_update")

        if self.vf_update_method == "pubsub":
            self.sdf_values = np.array(rospy.wait_for_message(sdf_update_topic, ValueFunctionMsg).vf).reshape(
                self.grid.shape
            )
        elif self.vf_update_method == "file":
            sdf_received = rospy.wait_for_message(sdf_update_topic, Bool).data
            self.sdf_values = np.array(np.load("./sdf.npy")).reshape(self.grid.shape)
        else:
            raise NotImplementedError("{} is not a valid vf update method".format(self.vf_update_method))

        self.brt = lambda sdf_values: lambda t, x: jnp.minimum(x, sdf_values)
        self.solver_settings = hj.SolverSettings.with_accuracy(
            self.vf_update_accuracy, value_postprocessor=self.brt(self.sdf_values)
        )

        self.vf_initialization_method = rospy.get_param("~vf_initialization_method")
        if self.vf_initialization_method == "sdf":
            self.vf = self.sdf_values.copy()
        elif self.vf_initialization_method == "cbf":
            cbf_params = rospy.get_param("/cbf")["Parameters"]
            original_cbf = QuadraticCBF(self.dynamics, cbf_params, test=False)
            tabular_cbf = TabularControlAffineCBF(self.dynamics, params={}, test=False, grid=self.grid)
            tabular_cbf.tabularize_cbf(original_cbf)
            self.vf = tabular_cbf.vf_table.copy()
        elif self.vf_initialization_method == "file":
            file_path = rospy.get_param("/vf_file")
            self.vf = np.load(file_path)
            if self.vf.ndim == self.grid.ndim + 1:
                self.vf = self.vf[-1]
            assert self.vf.shape == self.grid.shape, "vf file is not compatible with grid size"
        else:
            raise NotImplementedError("{} is not a valid initialization method".format(self.vf_initialization_method))

        # Set up value function publisher
        self.vf_topic = rospy.get_param("~topics/vf_update")

        if self.vf_update_method == "pubsub":
            self.vf_pub = rospy.Publisher(self.vf_topic, ValueFunctionMsg, queue_size=1)
        else:  # self.vf_update_method == "file":
            self.vf_pub = rospy.Publisher(self.vf_topic, Bool, queue_size=1)

        self.update_vf_flag = rospy.get_param("~update_vf_online")
        if not self.update_vf_flag:
            rospy.logwarn("Value function is not being updated")

        # Set up subscribers for disturbance, actuation, and obstacle updates
        disturbance_update_topic = rospy.get_param("~topics/disturbance_update")
        self.disturbance_update_sub = rospy.Subscriber(
            disturbance_update_topic, HiLoArray, self.callback_disturbance_update
        )

        actuation_update_topic = rospy.get_param("~topics/actuation_update")
        self.actuation_update_sub = rospy.Subscriber(actuation_update_topic, HiLoArray, self.callback_actuation_update)

        if self.vf_update_method == "pubsub":
            self.sdf_update_sub = rospy.Subscriber(
                sdf_update_topic, ValueFunctionMsg, self.callback_sdf_update_pubsub
            )
        else:  # self.vf_update_method == "file"
            self.sdf_update_sub = rospy.Subscriber(
                sdf_update_topic, Bool, self.callback_sdf_update_file
            )

        # Start updating the value function
        self.publish_initial_vf()
        self.update_vf()  # This keeps spinning

    def publish_initial_vf(self):
        while self.vf_pub.get_num_connections() != 2:
            rospy.loginfo("HJR node: Waiting for subscribers to connect")
            rospy.sleep(1)
        if self.vf_update_method == "pubsub":
            self.vf_pub.publish(ValueFunctionMsg(vf=self.vf.flatten()))
        else:  # self.vf_update_method == "file"
            np.save("./vf.npy", self.vf)
            self.vf_pub.publish(Bool(True))

    def callback_disturbance_update(self, msg):
        """
        Callback for the disturbance update subscriber.

        Args:
            msg (HiLoArray): The incoming disturbance update message.

        This method updates the disturbance space and the dynamics.
        """
        with self.vf_lock: 
            max_disturbance = msg.hi
            min_disturbance = msg.lo
            self.disturbance_space = hj.sets.Box(lo=jnp.array(min_disturbance), hi=jnp.array(max_disturbance))
            self.update_dynamics()  # FIXME:Check whether this is required or happens automatically

    def callback_actuation_update(self, msg):
        """
        Callback for the actuation update subscriber.

        Args:
            msg (HiLoArray): The incoming actuation update message.

        This method updates the control space and the dynamics.
        """
        with self.vf_lock:
            max_control = msg.hi
            min_control = msg.lo
            self.control_space = hj.sets.Box(lo=jnp.array(min_control), hi=jnp.array(max_control))
            self.update_dynamics()  # FIXME:Check whether this is required or happens automatically

    def callback_sdf_update_pubsub(self, msg):
        """
        Callback for the obstacle update subscriber.

        Args:
            msg (ValueFunctionMsg): The incoming obstacle update message.

        This method updates the obstacle and the solver settings.
        """
        with self.vf_lock:
            self.sdf_values = np.array(msg.vf).reshape(self.grid.shape)
            self.solver_settings = hj.SolverSettings.with_accuracy(
                self.vf_update_accuracy, value_postprocessor=self.brt(self.sdf_values)
            )

    def callback_sdf_update_file(self, msg):
        with self.vf_lock:
            if not msg.data:
                return
            self.sdf_values = np.array(np.load("./sdf.npy")).reshape(self.grid.shape)
            self.solver_settings = hj.SolverSettings.with_accuracy(
                self.vf_update_accuracy, value_postprocessor=self.brt(self.sdf_values)
            )
            rospy.loginfo("Processed SDF update")

    def update_dynamics(self):
        """
        Updates the Hamilton-Jacobi dynamics based on the current control and disturbance spaces.
        """
        self.hj_dynamics = HJControlAffineDynamics(
            self.dynamics,
            control_space=self.control_space,
            disturbance_space=self.disturbance_space,
        )

    def update_vf(self):
        """
        Continuously updates the value function and publishes it as long as the node is running and the update flag is set.
        """
        while not rospy.is_shutdown():
            if self.update_vf_flag:
                with self.vf_lock:
                    rospy.loginfo("Share of safe cells: {:.3f}".format(np.sum(self.vf >= 0) / self.vf.size))
                    time_now = rospy.Time.now().to_sec()
                    new_values = hj.step(
                        self.solver_settings,
                        self.hj_dynamics,
                        self.grid,
                        0.0,
                        self.vf.copy(),
                        -0.1,
                        progress_bar=False,
                    )
                    rospy.loginfo("Time taken to calculate vf: {:.2f}".format(rospy.Time.now().to_sec() - time_now))
                    self.vf = new_values
                if self.vf_update_method == "pubsub":
                    self.vf_pub.publish(ValueFunctionMsg(np.array(self.vf).flatten()))
                else:  # self.vf_update_method == "file"
                    np.save("./vf.npy", self.vf)
                    self.vf_pub.publish(Bool(True))

            rospy.sleep(0.05)  # To make sure that subscribers can run



if __name__ == "__main__":
    rospy.init_node("hj_reachability_node")
    HJReachabilityNode()
