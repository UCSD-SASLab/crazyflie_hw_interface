#!/usr/bin/env python3

import rospy
from refinecbf_ros.msg import Array

class BaseInterface:
    """
    BaseInterface is an abstract base class that converts the state and control messages 
    from the SafetyFilterNode to the correct type for the crazyflies. Each hardware platform 
    should have its own Interface node that subclasses this base class.

    Attributes:
    - state_msg_type: The ROS message type for the robot's state.
    - control_out_msg_type: The ROS message type for the robot's safe control.
    - external_control_msg_type: The ROS message type for the robot's external control.

    Subscribers:
    - ~topics/robot_state: Subscribes to the robot's state.
    - ~topics/cbf_safe_control: Subscribes to the safe control messages.
    - ~topics/robot_external_control: Subscribes to the external control messages.

    Publishers:
    - state_pub (~topics/cbf_state): Publishes the converted state messages.
    - safe_control_pub (~topics/robot_safe_control): Publishes the converted safe control messages.
    - external_control_pub (~topics/cbf_external_control): Publishes the converted external control messages.
    """

    state_msg_type = None
    control_out_msg_type = None
    external_control_msg_type = None
    disturbance_out_msg_type = None

    def __init__(self):
        """
        Initialize the BaseInterface.
        """
        # Set up state subscriber and publisher
        self.robot_state_topic = rospy.get_param("~topics/robot_state")
        cbf_state_topic = rospy.get_param("~topics/cbf_state")
        self.state_pub = rospy.Publisher(cbf_state_topic, Array, queue_size=1)

        # Set up safe control subscriber and publisher
        robot_safe_control_topic = rospy.get_param("~topics/robot_safe_control")
        self.cbf_safe_control_topic = rospy.get_param("~topics/cbf_safe_control")
        self.safe_control_pub = rospy.Publisher(robot_safe_control_topic, self.control_out_msg_type, queue_size=1)

        # Set up external control subscriber and publisher
        self.robot_external_control_topic = rospy.get_param("~topics/robot_external_control")
        cbf_external_control_topic = rospy.get_param("~topics/cbf_external_control")
        self.external_control_pub = rospy.Publisher(cbf_external_control_topic, Array, queue_size=1)

        # Set up disturbance subscriber and publisher
        if not rospy.get_param("~/env/disturbance_space/n_dims") == 0:
            robot_disturbance_topic = rospy.get_param("~topics/robot_disturbance")
            self.simulated_disturbance_topic = rospy.get_param("~topics/simulated_disturbance")
            self.disturbance_pub = rospy.Publisher(robot_disturbance_topic, self.disturbance_out_msg_type, queue_size=1)
        self.init_subscribers()

    def init_subscribers(self):
        rospy.Subscriber(self.robot_state_topic, self.state_msg_type, self.callback_state)
        rospy.Subscriber(self.cbf_safe_control_topic, Array, self.callback_safe_control)
        rospy.Subscriber(self.robot_external_control_topic, self.external_control_msg_type, self.callback_external_control)
        if not rospy.get_param("~/env/disturbance_space/n_dims") == 0:
            rospy.Subscriber(self.simulated_disturbance_topic, Array, self.callback_disturbance)


    def callback_state(self, state_msg):
        """
        Callback for the state subscriber. This method should be implemented in a subclass.

        Args:
            state_msg: The incoming state message.
        """
        raise NotImplementedError("Must be subclassed")

    def callback_safe_control(self, control_in_msg):
        """
        Callback for the safe control subscriber. This method should be implemented in a subclass.
        Should call self.override_safe_control()

        Args:
            control_msg: The incoming control message.
        """
        control_out_msg = self.process_safe_control(control_in_msg)
        assert type(control_out_msg) == self.control_out_msg_type, "Override to process the safe control message"
        if not self.override_safe_control():
            self.safe_control_pub.publish(control_out_msg)

    def callback_external_control(self, control_in_msg):
        """
        Callback for the external control subscriber. This method should be implemented in a subclass.
        Typical usage:
        - Process incoming control message to Array
        - Call self.override_nominal_control(control_msg)
        Args:
            control_msg: The incoming control message.
        """
        if self.override_safe_control():
            assert type(control_in_msg) == self.control_out_msg_type
            self.safe_control_pub.publish(control_in_msg)
        control_out_msg = self.process_external_control(control_in_msg)
        assert type(control_out_msg) == Array, "Override to process the external control message"
        if self.override_nominal_control():
            self.external_control_pub.publish(control_out_msg) if self.override_nominal_control() else None

    def callback_disturbance(self, disturbance_msg):
        disturbance_out_msg = self.process_disturbance(disturbance_msg)
        assert type(disturbance_out_msg) == self.disturbance_out_msg_type, "Override to process the disturbance message"
        self.disturbance_pub.publish(disturbance_out_msg)

    def process_external_control(self, control_in_msg):
        raise NotImplementedError("Must be subclassed")
    
    def process_safe_control(self, control_in_msg):
        raise NotImplementedError("Must be subclassed")
    
    def process_disturbance(self, disturbance_msg):
        raise NotImplementedError("Must be subclassed")

    def clip_control_output(self, control_in_msg):
        return control_in_msg
    
    def override_safe_control(self):
        """
        Checks if the robot should override the safe control. Defaults to False.
        Should be overriden if the robot has to be able to be taken over by user.

        Returns:
            True if the robot should override the safe control, False otherwise.
        """
        return False
    
    def override_nominal_control(self):
        """
        Checks if the robot should override the nominal control. Defaults to False.
        Should be overriden if we would like interactive experiments. (e.g. geofencing)

        Returns:
            True if the robot should override the nominal control, False otherwise.
        """
        return False