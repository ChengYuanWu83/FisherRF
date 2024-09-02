import rospy
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from . import utils
import numpy as np
import os
import csv
import math
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import time
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class SimulatorBridge:
    def __init__(self, cfg):
        self.cv_bridge = CvBridge()
        self.current_rgb = None
        self.current_depth = None
        self.current_pose = None
        self.camera_type = cfg["camera_type"]
        self.sensor_noise = cfg["sensor_noise"]
        # self.command_pose = None

        self.get_simulator_camera_info()
        #[cyw]:change pose_pub which publish camera_info to publish uav pose
        """ 
        self.pose_pub = rospy.Publisher(
            "/gazebo/set_model_state", ModelState, queue_size=1, latch=True
        )
        """
        #[cyw]:uavpose topic "/firefly/ground_truth/pose" ,camera_pose topic "/firefly/transform_stamped"
        self.current_pose_sub = rospy.Subscriber(
            "/firefly/transform_stamped", TransformStamped, self.update_current_pose
        )
        # self.current_pose_sub = rospy.Subscriber(
        #     "/firefly/ground_truth/pose", Pose, self.update_current_pose
        # )

        self.pose_pub = rospy.Publisher(
            "/nbv/uav_pose", PoseStamped, queue_size=1, latch=True
        )

        if self.camera_type == "rgb_camera":
            self.rgb_sub = rospy.Subscriber(
                "/rgb_camera/rgb_image_raw", Image, self.update_rgb
            )
        elif self.camera_type == "rgbd_camera":
            self.rgb_sub = rospy.Subscriber(
                "/rgbd_camera/rgb_image_raw", Image, self.update_rgb
            )
            self.depth_sub = rospy.Subscriber(
                "/rgbd_camera/depth_image_raw", Image, self.update_depth
            )
        elif self.camera_type == "firefly/realsense":
            self.rgb_sub = rospy.Subscriber(
                f"/{self.camera_type}/rgb/image_raw", Image, self.update_rgb
            )
            self.depth_sub = rospy.Subscriber(
                f"/{self.camera_type}/depth/image_raw", Image, self.update_depth
            )
        elif self.camera_type == "uav/camera/left":
            self.rgb_sub = rospy.Subscriber(
                f"/{self.camera_type}/image_rect_color", Image, self.update_rgb
            )
        else : #[cyw]: add rotorS image sub
            self.rgb_sub = rospy.Subscriber(
                f"/{self.camera_type}/image_raw", Image, self.update_rgb
            )
            

    def get_simulator_camera_info(self):
        if self.camera_type == "uav/camera/left":
            camera_info_raw = rospy.wait_for_message(
                f"/{self.camera_type}/camera_info", CameraInfo
            )
        else:
            camera_info_raw = rospy.wait_for_message(
                f"/{self.camera_type}/rgb/camera_info", CameraInfo
            )
        K = camera_info_raw.K  # intrinsic matrix
        H = int(camera_info_raw.height)  # image height
        W = int(camera_info_raw.width)  # image width
        self.camera_info = {
            "image_resolution": [H, W],
            "c": [K[2], K[5]],
            "focal": [K[0], K[4]],
        }

    def move_camera(self, pose): 
        quaternion = utils.rotation_2_quaternion(pose[:3, :3])
        translation = pose[:3, -1]

        camera_pose_msg = ModelState()
        camera_pose_msg.model_name = self.camera_type
        camera_pose_msg.pose.position.x = translation[0]
        camera_pose_msg.pose.position.y = translation[1]
        camera_pose_msg.pose.position.z = translation[2]
        camera_pose_msg.pose.orientation.x = quaternion[0]
        camera_pose_msg.pose.orientation.y = quaternion[1]
        camera_pose_msg.pose.orientation.z = quaternion[2]
        camera_pose_msg.pose.orientation.w = quaternion[3]

        self.pose_pub.publish(camera_pose_msg)
    #[cyw]:unpack the ros pose datastructure
    def get_current_ros_pose_in_list(self):
        if ( self.current_pose is None):
            print("wait for pose")
            time.sleep(0.01)
        if isinstance(self.current_pose, TransformStamped):
            translation = self.current_pose.transform.translation
            rotation = self.current_pose.transform.rotation
            current_position = [translation.x, translation.y, translation.z]
            current_orientation = [rotation.x, rotation.y, rotation.z, rotation.w]
        elif isinstance(self.current_pose, Pose):
            current_position = [self.current_pose.position.x,
                        self.current_pose.position.y,
                        self.current_pose.position.z]
            current_orientation = [self.current_pose.orientation.x,
                        self.current_pose.orientation.y,
                        self.current_pose.orientation.z,
                        self.current_pose.orientation.w]
        elif self.current_pose is None:
            print("isnone")
            return self.get_current_ros_pose_in_list()
        else:
            print("don't know")
        return current_position, current_orientation
    #[cyw]: move uav func
    def slow_move_uav_in_rotations(self, desired_pose):
        new_desired_pose = copy.deepcopy(desired_pose)
        current_position, current_orientation = self.get_current_ros_pose_in_list()

        # interpolate on position
        current_position = np.array(current_position)
        desired_position = np.array(new_desired_pose[:3])
        distance = np.linalg.norm(desired_position - current_position)

        # print(f"current_position: {current_position}")

        if distance > 1:
            unit = (desired_position - current_position) / distance / 1.5
            new_desired_pose[:3] = unit + current_position

            #interpolate on orientation
            r1 = R.from_quat(current_orientation)
            r2 = R.from_quat(new_desired_pose[3:])
            key_times = [0, 1]
            key_rots = R.from_quat([r1.as_quat(), r2.as_quat()])
            slerped_rots = Slerp(key_times, key_rots)

            times = np.linspace(0, 1, int(distance) + 1)
            interpolated_quaternions = slerped_rots(times)
            new_desired_pose[3:] = interpolated_quaternions[1].as_quat()
            
        self.move_uav(new_desired_pose)
        # return uav_pose_msg

    def move_uav(self, pose): 
        # quaternion = utils.rotation_2_quaternion(pose[:3, :3])
        # translation = pose[:3, -1]


        uav_pose_msg = PoseStamped()
        # uav_pose_msg.model_name = self.camera_type
        uav_pose_msg.pose.position.x = pose[0]
        uav_pose_msg.pose.position.y = pose[1]
        uav_pose_msg.pose.position.z = pose[2]
        uav_pose_msg.pose.orientation.x = pose[3]
        uav_pose_msg.pose.orientation.y = pose[4]
        uav_pose_msg.pose.orientation.z = pose[5]
        uav_pose_msg.pose.orientation.w = pose[6]
        uav_pose_msg.header.stamp = rospy.Time.now()

        # self.command_pose = uav_pose_msg

        # print(uav_pose_msg)
        #[cyw]: Save a pose to a CSV file.
        
        self.pose_pub.publish(uav_pose_msg)
        # return uav_pose_msg
    
    def slow_move_uav(self, desired_pose):
        new_desired_pose = copy.deepcopy(desired_pose)
        current_position, current_orientation = self.get_current_ros_pose_in_list()

        current_position = np.array(current_position)
        desired_position = np.array(new_desired_pose[:3])
        distance = np.linalg.norm(desired_position - current_position)
        if distance > 1:
            unit = (desired_position - current_position) / distance / 2
            new_desired_pose[:3] = unit + current_position
        # print(f"current_position: {current_position}")
        self.move_uav(new_desired_pose)

    def update_rgb(self, data):
        self.current_rgb = data

    def update_depth(self, data):
        self.current_depth = data

    def update_current_pose(self, data): #[cyw]
        self.current_pose = data

    def get_image(self): # [cyw]   
        # the camera pose
        captured_pose = self.current_pose    
        if self.current_rgb is None:
            print("wait for image")
            time.sleep(0.01)


        rgb = self.cv_bridge.imgmsg_to_cv2(self.current_rgb, "rgb8")
        # rgb = np.array(rgb, dtype=float)

        # if self.sensor_noise != 0:
        #     noise = np.random.normal(0.0, self.sensor_noise, rgb.shape)
        #     rgb += noise

        if self.camera_type == "rgb_camera":
            depth = None
        elif self.camera_type == "rgbd_camera":
            depth = self.cv_bridge.imgmsg_to_cv2(self.current_depth, "32FC1")
        else :
            depth = None

        return np.asarray(rgb), np.asarray(depth), captured_pose
    #[cyw]:
    def check_if_uav_arrive(self, desired_pose):
        # Calculate the distance to the desired pose
        current_position, current_orientation = self.get_current_ros_pose_in_list()

        current_pose = [*current_position, *current_orientation]    
        distance = math.sqrt(
            (desired_pose[0] - current_pose[0]) ** 2 +
            (desired_pose[1] - current_pose[1]) ** 2 +
            (desired_pose[2] - current_pose[2]) ** 2   )
            
        current_orientation = current_pose[-4:]
        desired_orientation = desired_pose[-4:]
        current_euler = euler_from_quaternion(current_orientation)
        desired_euler = euler_from_quaternion(desired_orientation)

        orientation_diff = math.sqrt(
            (current_euler[0] - desired_euler[0]) ** 2 +
            (current_euler[1] - desired_euler[1]) ** 2 +
            (current_euler[2] - desired_euler[2]) ** 2
        )
        # print(f"desired_pose UAV Pose: {desired_pose}")
        # print(f"current_pose UAV Pose: x={self.current_pose.position.x}, y={self.current_pose.position.y}, z={self.current_pose.position.z}")
        # print(f"Distance to desired pose: {distance}, and orientation_diff: {orientation_diff}")

        # Check if the UAV has reached the desired pose
        if distance < 0.5 and orientation_diff < 10:  # You can adjust the threshold
            #self.reach_publisher.publish("reach")
            print(f"Distance to desired pose: {distance}, and orientation_diff: {orientation_diff}")
            # print(f"UAV has reached the desired pose {desired_pose}, current pose {current_pose}")
            return 1
        return 0