#!/usr/bin/env python3

import rospy
import tf
from geometry_msgs.msg import Twist
import rowan


if __name__ == "__main__":
    rospy.init_node('tf_to_sim')
    listener = tf.TransformListener()
    publisher = rospy.Publisher('/sim_state', Twist, queue_size=1)
    rate = rospy.Rate(100.0)

    while not rospy.is_shutdown():
        try:
            (trans, rot) = listener.lookupTransform('/odom', '/base_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        cmd = Twist()
        cmd.linear.x = trans[0]
        cmd.linear.y = trans[1]
        cmd.linear.z = trans[2]
        angles = rowan.to_euler(rot)
        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = angles[2]
        publisher.publish(cmd)

