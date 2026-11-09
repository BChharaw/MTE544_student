#usage: bash run.sh [spiral|point|trajectory]   (defaults to spiral)
MOTION=${1:-spiral}

echo "domain id: $ROS_DOMAIN_ID" #should be 6, if blank you forgot to source/export

#kill leftover runs so two nodes arent fighting over /cmd_vel
pkill -f "decisions.py" && echo "killed an old decisions.py" && sleep 1

#send one zero cmd on ctrl+c so the robot actually stops
stop_robot(){
    echo "stopping robot..."
    timeout 5 ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}" >/dev/null
}
trap stop_robot INT

#warn if something else is already publishing cmd_vel (teleop, another group, etc)
PUBS=$(timeout 10 ros2 topic info /cmd_vel 2>/dev/null | grep "Publisher count" | awk '{print $3}')
if [ -n "$PUBS" ] && [ "$PUBS" -gt 0 ]; then
    echo "WARNING: $PUBS other publisher(s) on /cmd_vel, close teleop/other runs or robot will stop-start"
fi

echo "resetting odom..."
timeout 15 ros2 service call /reset_pose irobot_create_msgs/srv/ResetPose || echo "reset timed out, check vpn/domain id"

echo "starting $MOTION..."
python3 decisions.py --motion $MOTION
