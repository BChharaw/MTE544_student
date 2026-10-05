# Imports
import rclpy

from rclpy.node import Node

from utilities import Logger, euler_from_quaternion
from rclpy.qos import QoSProfile

# referenced the messaged types for the imports
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from rclpy.time import Time

# needed for the qos profile
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


CIRCLE=0; SPIRAL=1; ACC_LINE=2
motion_types=['circle', 'spiral', 'line']

# Max velocity and angular velocity for the robot
MAX_VELOCITY= 0.5
MAX_ANGULAR_VELOCITY= 2.0
class motion_executioner(Node):
    
    def __init__(self, motion_type=0):
        
        super().__init__("motion_types")
        
        self.type=motion_type
        
        self.radius_= 1.5
        self.prev_velocity = 0.5
        self.successful_init=False
        self.imu_initialized=False
        self.odom_initialized=False
        self.laser_initialized=False
        qos=QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)      
        self.vel_publisher=self.create_publisher(Twist, 'cmd_vel', qos)
                
        # loggers
        self.imu_logger=Logger('imu_content_'+str(motion_types[motion_type])+'.csv', headers=["acc_x", "acc_y", "angular_z", "stamp"])
        self.odom_logger=Logger('odom_content_'+str(motion_types[motion_type])+'.csv', headers=["x","y","th", "stamp"])
        self.laser_logger=Logger('laser_content_'+str(motion_types[motion_type])+'.csv', headers=["ranges", "angle_increment", "stamp"])

        # added in qos profile + subribers
        qos=QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)        
        self.imu_subscription=self.create_subscription(Imu, 'imu', self.imu_callback, qos)
        self.encoder_subscription=self.create_subscription(Odometry, 'odom', self.odom_callback, qos)
        self.laser_subscription=self.create_subscription(LaserScan, 'scan', self.laser_callback, qos)
        
        
        self.create_timer(0.1, self.timer_callback)


    def imu_callback(self, imu_msg: Imu):
        acc_x=imu_msg.linear_acceleration.x
        acc_y=imu_msg.linear_acceleration.y
        angular_z=imu_msg.angular_velocity.z
        stamp=Time.from_msg(imu_msg.header.stamp).nanoseconds

        self.imu_logger.log_values([acc_x, acc_y, angular_z, stamp])
        self.imu_initialized=True
        
    def odom_callback(self, odom_msg: Odometry):
        x=odom_msg.pose.pose.position.x
        y=odom_msg.pose.pose.position.y
        q=odom_msg.pose.pose.orientation
        th=euler_from_quaternion([q.x, q.y, q.z, q.w])
        stamp=Time.from_msg(odom_msg.header.stamp).nanoseconds

        self.odom_logger.log_values([x, y, th, stamp])
        self.odom_initialized=True

    def laser_callback(self, laser_msg: LaserScan):
        ranges=list(laser_msg.ranges)
        angle_increment=laser_msg.angle_increment
        stamp=Time.from_msg(laser_msg.header.stamp).nanoseconds

        self.laser_logger.log_values(ranges+[angle_increment, stamp])
        self.laser_initialized=True

    def timer_callback(self):
        
        if self.odom_initialized and self.laser_initialized and self.imu_initialized:
            self.successful_init=True
            
        if not self.successful_init:
            return
        
        cmd_vel_msg=Twist()
        
        if self.type==CIRCLE:
            cmd_vel_msg=self.make_circular_twist()
        
        elif self.type==SPIRAL:
            cmd_vel_msg=self.make_spiral_twist()
                        
        elif self.type==ACC_LINE:
            cmd_vel_msg=self.make_acc_line_twist()
            
        else:
            print("type not set successfully, 0: CIRCLE 1: SPIRAL and 2: ACCELERATED LINE")
            raise SystemExit 

        # make sure the velocity and angular velocity are within the limits
        if cmd_vel_msg.angular.z > MAX_ANGULAR_VELOCITY:
            cmd_vel_msg.angular.z = MAX_ANGULAR_VELOCITY
        if cmd_vel_msg.linear.x > MAX_VELOCITY:
            cmd_vel_msg.linear.x = MAX_VELOCITY
            self.prev_velocity = MAX_VELOCITY
        elif cmd_vel_msg.linear.x < 0:
            cmd_vel_msg.linear.x = 0.0
        
        # publish the command velocity message
        self.vel_publisher.publish(cmd_vel_msg)
        
    

    def make_circular_twist(self):
        msg=Twist()
        # Add linear and angular velocity to the message
        msg.linear.x = 0.2
        msg.angular.z = MAX_VELOCITY/self.radius_
        return msg

    def make_spiral_twist(self):
        msg=Twist()
        self.radius_ -= 0.005
        msg.linear.x = self.radius_
        msg.angular.z = MAX_ANGULAR_VELOCITY
        return msg
    
    def make_acc_line_twist(self):
        msg=Twist()
        self.prev_velocity += 0.1
        msg.angular.z = 0.0
        msg.linear.x = self.prev_velocity
        return msg

import argparse

if __name__=="__main__":
    

    argParser=argparse.ArgumentParser(description="input the motion type")


    argParser.add_argument("--motion", type=str, default="circle")



    rclpy.init()

    args = argParser.parse_args()

    if args.motion.lower() == "circle":

        ME=motion_executioner(motion_type=CIRCLE)
    elif args.motion.lower() == "line":
        ME=motion_executioner(motion_type=ACC_LINE)

    elif args.motion.lower() =="spiral":
        ME=motion_executioner(motion_type=SPIRAL)

    else:
        print(f"we don't have {arg.motion.lower()} motion type")


    
    try:
        rclpy.spin(ME)
    except KeyboardInterrupt:
        print("Exiting")
