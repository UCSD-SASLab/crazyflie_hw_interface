#!/usr/bin/env python3

import rospy
import numpy as np
import jax.numpy as jnp
import jax
import hj_reachability as hj
from refinecbf_ros.msg import Array,ValueFunctionMsg
from refinecbf_ros.config import Config
from refinecbf_ros.srv import ActivateObstacle, ActivateObstacleResponse
import pdb
import matplotlib.pyplot as plt

class ObstacleNode:

    def __init__(self) -> None:
        # Following publishers:
        # - /env/obstacle_update
        # Following subscribers:
        # - /state

        # Config:
        config = Config(hj_setup=True)
        self.dynamics = config.dynamics
        self.grid = config.grid
        self.detection_obstacles = config.detection_obstacles
        self.service_obstacles = config.service_obstacles
        self.update_obstacles = config.update_obstacles
        self.boundary = config.boundary
        
        # Publishers:
        obstacle_update_topic = rospy.get_param("~topics/obstacle_update", "/env/obstacle_update")
        self.obstacle_update_pub = rospy.Publisher(obstacle_update_topic, ValueFunctionMsg, queue_size=1)

        # Subscribers:
        cbf_state_topic = rospy.get_param("~topics/cbf_state")
        state_sub = rospy.Subscriber(cbf_state_topic, Array, self.callback_state)

        # Services:
        activate_obstacle_service = rospy.get_param("~services/activate_obstacle")
        rospy.Service(activate_obstacle_service,ActivateObstacle,self.handle_activate_obstacle)

        # Initialize Active Obstacles (Just Boundary):
        self.active_obstacles = []
        #breakpoint()
        self.update_sdf()

        #fig = plt.figure(figsize=(6, 5))
        #f = plt.contourf(self.grid.coordinate_vectors[0], self.grid.coordinate_vectors[1], vf[:, :, self.grid.shape[2] // 2])
        #plt.contour(self.grid.coordinate_vectors[0], self.grid.coordinate_vectors[1],vf[:, :, self.grid.shape[2] // 2].T, levels=[0], colors='k')
        #plt.colorbar(f)
        #plt.show()

        #breakpoint()



    def obstacle_detection(self):
        updatesdf = False
        for obstacle in self.detection_obstacles:
            if obstacle not in self.active_obstacles:
                if obstacle.distance_to_obstacle(self.robot_state) <= obstacle.detectionRadius:
                    self.active_obstacles.append(obstacle)
                    updatesdf = True
        for obstacle in self.update_obstacles: 
            if obstacle not in self.active_obstacles:
                if obstacle.updateTime >= rospy.Time.now().to_sec():
                    self.active_obstacles.append(obstacle)
                    updatesdf = True
        
        if updatesdf:
            self.update_sdf()

    def update_sdf(self):
        sdf_msg = ValueFunctionMsg()
        sdf_msg.vf = hj.utils.multivmap(self.build_sdf(), jnp.arange(self.grid.ndim))(self.grid.states)
        self.obstacle_update_pub.publish(sdf_msg)    

    def callback_state(self, state_msg):
        self.robot_state = jnp.reshape(np.array(state_msg.value),(-1,1))

    def build_sdf(self):
        def sdf(x):
            sdf = self.boundary.boundary_sdf(x)
            for obstacle in self.active_obstacles:
                obstacle_sdf = obstacle.obstacle_sdf(x)
                sdf = jnp.min(jnp.array([sdf, obstacle_sdf]))
            return sdf
        return sdf

    def handle_activate_obstacle(self,req):
        obstacle_index = req.obstacleNumber
        if obstacle_index >= len(self.service_obstacles):
            output = "Invalid Obstacle Number"
        elif self.service_obstacles[obstacle_index] in self.active_obstacles:
            output = "Obstacle Already Active"
        else:
            self.active_obstacles.append(self.service_obstacles[obstacle_index])
            self.update_sdf()
            output = "Obstacle Activated"
        return(ActivateObstacleResponse(output))


if __name__ == "__main__":
    rospy.init_node("obstacle_node")
    ObstacleNodeObject = ObstacleNode()

    rate = rospy.Rate(rospy.get_param("~/env/obstacle_update_rate_hz"))

    while not rospy.is_shutdown():
        ObstacleNodeObject.obstacle_detection()
        rate.sleep()