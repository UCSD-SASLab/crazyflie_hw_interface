#!/usr/bin/env python3

import rospy
import numpy as np
import hj_reachability as hj
import jax.numpy as jnp

from refinecbf_ros.config import Config

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jackal.hjr_nominal_control import NominalControlHJ
from template.nominal_controller import NominalController

class JackalNominalControl(NominalController):
    def __init__(self):
        super().__init__()
        self.config = Config(hj_setup=True)
        self.dynamics = self.config.dynamics
        self.hj_dynamics = self.config.hj_dynamics

        self.state = np.zeros(self.dynamics.n_dims)
        self.grid = self.config.grid

        self.target = jnp.array(rospy.get_param("/ctr/nominal/goal/coordinates"))
        assert len(self.target) == self.grid.ndim
        self.padding = jnp.array(rospy.get_param("/ctr/nominal/goal/padding"))
        assert len(self.padding) == self.grid.ndim
        self.max_time = rospy.get_param("/ctr/nominal/goal/max_time")
        assert self.max_time > 0
        self.time_intervals = int(rospy.get_param("/ctr/nominal/goal/time_intervals"))
        assert self.time_intervals > 0
        self.solver_accuracy = rospy.get_param("/ctr/nominal/goal/solver_accuracy", "low")
        assert self.solver_accuracy in ["low", "medium", "high", "very_high"]

        # Initialize Controller
        self.controller_prep = NominalControlHJ(
            self.hj_dynamics,
            self.grid,
            final_time=self.max_time,
            time_intervals=self.time_intervals,
            solver_accuracy=self.solver_accuracy,
            target=self.target,
            padding=self.padding,
        )
        rospy.loginfo("Solving for nominal control, nominal control default is 0")
        self.controller = lambda x, t: np.zeros(self.dynamics.control_dims)
        self.controller_prep.solve()  # Solves the problem so that it becomes table lookup
        self.controller = self.controller_prep.get_nominal_control

        rospy.loginfo("Nominal controller ready!")


if __name__ == "__main__":
    rospy.init_node("jackal_nominal_control_node")
    Controller = JackalNominalControl()

    rate = rospy.Rate(rospy.get_param("/ctr/nominal/frequency"))

    while not rospy.is_shutdown():
        Controller.publish_control()
        rate.sleep()
