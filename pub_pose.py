
import rospy

from geometry_msgs.msg import PoseStamped


def move_uav(): 
    pub = rospy.Publisher("/nbv/uav_pose", PoseStamped, queue_size=1, latch=True)
    rospy.init_node('adf', anonymous=True)
    #print('start publise pose')
    rate = rospy.Rate(10)


    while not rospy.is_shutdown():
        uav_pose_msg = PoseStamped()
        # uav_pose_msg.model_name = self.camera_type
        uav_pose_msg.pose.position.x = 0.0
        uav_pose_msg.pose.position.y = 0.0
        uav_pose_msg.pose.position.z = 1.0
        uav_pose_msg.pose.orientation.x = 0.0
        uav_pose_msg.pose.orientation.y = 0.0
        uav_pose_msg.pose.orientation.z = 1.0
        uav_pose_msg.pose.orientation.w = 0.0
        uav_pose_msg.header.stamp = rospy.Time.now()

        print('start publise pose')
        pub.publish(uav_pose_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        move_uav()
    except rospy.ROSInterruptException:
        pass