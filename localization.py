import sys

from utilities import Logger, euler_from_quaternion
from rclpy.time import Time
from rclpy.node import Node

from rclpy.qos import QoSProfile,ReliabilityPolicy,HistoryPolicy
from nav_msgs.msg import Odometry as odom

from rclpy import init, spin

rawSensor = 0
class localization(Node):
    
    def __init__(self, localizationType=rawSensor):

        super().__init__("localizer")
        
        # Remember to define your QoS profile based on the information available in "ros2 topic info /odom --verbose" as explained in Tutorial 3

        odom_qos=QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST,depth=1)
        
        self.loc_logger=Logger("robot_pose.csv", ["x", "y", "theta", "stamp"])
        self.pose=None
        
        if localizationType == rawSensor:
        #subscribe to wheel odometry
            self.create_subscription(odom,'odom',self.odom_callback,odom_qos)
        else:
            print("This type doesn't exist", sys.stderr)
    
    
    def odom_callback(self, pose_msg):
        
        #pose comes from the orientation quaternion
        x=pose_msg.pose.pose.position.x
        y=pose_msg.pose.pose.position.y
        q=pose_msg.pose.pose.orientation
        th=euler_from_quaternion([q.x,q.y,q.z,q.w])
        self.pose=[x,y,th,pose_msg.header.stamp]
        
        # Log the data
        self.loc_logger.log_values([self.pose[0], self.pose[1], self.pose[2], Time.from_msg(self.pose[3]).nanoseconds])
    
    def getPose(self):
        return self.pose

# Here put a guard that makes the node run, ONLY when run as a main thread!
# This is to make sure this node functions right before using it in decision.py

#generic boilerplate for if main
if __name__=="__main__":
    init()
    LOC=localization()
    spin(LOC)
