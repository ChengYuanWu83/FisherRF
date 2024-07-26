
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from planner import utils
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def change_pose(i):
    #define view
    # phi = i * (np.pi/180)
    # theta = 0
    phi = 0
    theta = i * (np.pi/180)

    # phi = 45 * (np.pi/180)
    # theta = 45 * (np.pi/180)
    # radius = 3
    # x = radius * np.cos(theta) * np.cos(phi)
    # y = radius * np.sin(theta) * np.cos(phi)
    # z = radius * np.sin(phi)


    print(f"phi: {phi * (180/np.pi)}, theta: {theta * (180/np.pi)}")
    #calculate translation
    pose = np.eye(4)

    x = 3 
    y = -3
    z = 1
    translation = np.array([x, y, z])
    #calculate rotation matrix
    # rotation_matrix = R.from_euler("ZYX", [theta, np.pi+phi, np.pi]).as_matrix()
    opengl_pose = utils.points_to_pose(translation)
    rotation_matrix = opengl_pose[:3,:3]
    quaternion = utils.rotation_2_quaternion(rotation_matrix)
    
    return translation, quaternion

def move_uav(): 
    pub = rospy.Publisher("/nbv/uav_pose", PoseStamped, queue_size=1, latch=True)
    rospy.init_node('test_pub_pose', anonymous=True)
    #print('start publise pose')
    rate = rospy.Rate(10)


    i = 0
    n=20
    translation, quaternion = change_pose(i)
    while not rospy.is_shutdown():
        i += 1
        if i % n == 0:
            translation, quaternion = change_pose(i)

        
        uav_pose_msg = PoseStamped()
        # uav_pose_msg.model_name = self.camera_type
        uav_pose_msg.pose.position.x = translation[0]
        uav_pose_msg.pose.position.y = translation[1]
        uav_pose_msg.pose.position.z = translation[2]
        uav_pose_msg.pose.orientation.x = quaternion[0]
        uav_pose_msg.pose.orientation.y = quaternion[1]
        uav_pose_msg.pose.orientation.z = quaternion[2]
        uav_pose_msg.pose.orientation.w = quaternion[3]
        uav_pose_msg.header.stamp = rospy.Time.now()
        
        if i % n == 0:
            print(uav_pose_msg)
        # print('start publise pose')
        pub.publish(uav_pose_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        move_uav()
    except rospy.ROSInterruptException:
        pass