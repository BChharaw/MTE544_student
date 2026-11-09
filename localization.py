import sys

from utilities import Logger

from rclpy.time import Time

from utilities import euler_from_quaternion, calculate_angular_error, calculate_linear_error
from rclpy.node import Node
from geometry_msgs.msg import Twist

from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry as odom

from sensor_msgs.msg import Imu
from kalman_filter import kalman_filter

from rclpy import init, spin, spin_once

import numpy as np
import message_filters

rawSensors=0
kalmanFilter=1
odom_qos=QoSProfile(reliability=2, durability=2, history=1, depth=10)

class localization(Node):
    
    def __init__(self, type, dt, loggerName="robotPose.csv", loggerHeaders=["imu_ax", "imu_ay", "kf_ax", "kf_ay","kf_vx","kf_w","kf_x", "kf_y","stamp"]):

        super().__init__("localizer")

        self.loc_logger=Logger( loggerName , loggerHeaders)
        self.pose=None
        
        if type==rawSensors:
            self.initRawSensors()
        elif type==kalmanFilter:
            self.initKalmanfilter(dt)
        else:
            print("We don't have this type for localization", sys.stderr)
            return  

    def initRawSensors(self):
        self.create_subscription(odom, "/odom", self.odom_callback, qos_profile=odom_qos)
        
    def initKalmanfilter(self, dt):
        
        x=np.zeros(6) #state, robot starts at the origin wiht everything zero
        Q=0.5*np.eye(6) #state noise, (0.5 from readme)
        R=0.15*np.eye(4) #measurement noise (also 0.5 from readme)
        P=np.eye(6) #initial cov matrix
        
        self.kf=kalman_filter(P,Q,R,x,dt)
        
        self.odom_sub=message_filters.Subscriber(self,odom,"/odom",qos_profile=odom_qos) #v w come from odom
        self.imu_sub=message_filters.Subscriber(self,Imu,"/imu",qos_profile=odom_qos) #ax ay come from the imu
        
        time_syncher=message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.imu_sub], queue_size=10, slop=0.1)
        time_syncher.registerCallback(self.fusion_callback)
    
    def fusion_callback(self, odom_msg: odom, imu_msg: Imu):
        
        #measurements
        z=np.array([odom_msg.twist.twist.linear.x, odom_msg.twist.twist.angular.z, imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y]) 
        
        self.kf.predict() #step 1 predict via motion model
        self.kf.update(z) #step 2 correct the prediction with measurements
        
        xhat=self.kf.get_states()

        self.pose=np.array([xhat[0],xhat[1],xhat[2],odom_msg.header.stamp]) #set pose

        self.loc_logger.log_values([z[2],z[3],xhat[5],xhat[4]*xhat[3],xhat[4],xhat[3],xhat[0],xhat[1],Time.from_msg(odom_msg.header.stamp).nanoseconds]) #log state and measurements
      
    def odom_callback(self, pose_msg):
        
        self.pose=[ pose_msg.pose.pose.position.x,
                    pose_msg.pose.pose.position.y,
                    euler_from_quaternion(pose_msg.pose.pose.orientation),
                    pose_msg.header.stamp]

    # Return the estimated pose
    def getPose(self):
        return self.pose


if __name__=="__main__":
    
    init()
    
    LOCALIZER=localization(kalmanFilter, 0.1)
    
    spin(LOCALIZER)
