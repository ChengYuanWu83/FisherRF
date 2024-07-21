
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped
from planner.utils import view_to_pose, rotation_2_quaternion
from scipy.spatial.transform import Rotation as R

current_pose = None
def update_current_pose(data): #[cyw]
    current_pose = data

def move_uav(): 
    current_pose_sub = rospy.Subscriber(
            "/firefly/transform_stamped", TransformStamped, update_current_pose
        )
    rospy.init_node('test_pub_pose', anonymous=True)
    #print('start publise pose')
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        if current_pose is not None:
            # Do something with current_pose
            # For example, convert TransformStamped to PoseStamped and publish it
            pose_msg = PoseStamped()
            pose_msg.header = current_pose.header
            pose_msg.pose.position.x = current_pose.transform.translation.x
            pose_msg.pose.position.y = current_pose.transform.translation.y
            pose_msg.pose.position.z = current_pose.transform.translation.z
            
            rotation = R.from_quat([
                current_pose.transform.rotation.x,
                current_pose.transform.rotation.y,
                current_pose.transform.rotation.z,
                current_pose.transform.rotation.w
            ])
            quat = rotation_2_quaternion(rotation)
            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]
            
            # You can publish the pose_msg if needed
            # pose_pub.publish(pose_msg)
            
            rospy.loginfo("Current pose: %s", pose_msg)
            print(current_pose)
        rate.sleep()


if __name__ == '__main__':
    try:
        move_uav()
    except rospy.ROSInterruptException:
        pass