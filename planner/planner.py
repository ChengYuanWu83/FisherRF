from .simulator_bridge import SimulatorBridge
from . import utils
import time
import os
from datetime import datetime
import numpy as np
import imageio.v2 as imageio
import yaml
import json
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped
from scipy.spatial.transform import Rotation as R
import csv

class Planner:
    def __init__(self, cfg):
        #[cyw]: my parameter
        self.planning_time = 0
        self.radius_start = cfg["radius_start"]
        self.radius_end = cfg["radius_end"]
        self.sampling_method = cfg["sampling_method"]

        self.simulator_bridge = SimulatorBridge(cfg["simulation_bridge"])
        self.camera_info = self.simulator_bridge.camera_info

        # self.record_path = os.path.join(
        #     cfg["experiment_path"], cfg["planner_type"], str(cfg["experiment_id"])
        # )
        self.record_path = cfg["experiment_path"]

        self.planning_budget = 20
        self.initial_type = cfg["initial_type"]

        self.H, self.W = self.camera_info[
            "image_resolution"
        ]  # original image resolution from simulator
        self.trajectory = np.empty((self.planning_budget, 4, 4))
        self.view_trajectory = np.empty((self.planning_budget, 2))  # [phi, theta]
        self.rgb_measurements = np.empty((self.planning_budget, self.H, self.W, 3))
        self.depth_measurements = np.empty((self.planning_budget, self.H, self.W))
        self.step = 0

        self.config_actionspace(cfg["action_space"])

    def config_actionspace(self, cfg):
        """set hemisphere actionspace parameters"""

        self.min_height = cfg["min_height"]
        self.radius = cfg["radius"]
        # self.phi_min = np.arcsin(self.min_height / self.radius)
        self.phi_min = cfg["phi_min"]
        self.phi_max = 0.5 * np.pi
        self.theta_min = 0
        self.theta_max = 2 * np.pi

    def init_camera_pose(self, initial_view):
        print("------ start mission ------ \n")
        print("------ initialize camera pose ------ \n")

        if initial_view is None:
            if self.initial_type == "random":
                initial_view = utils.uniform_sampling(self.radius, self.phi_min)

            elif self.initial_type == "pre_calculated":
                self.get_view_list()
                initial_view = next(self.view_list)

            self.move_sensor(initial_view)

        else:
            for view in initial_view:
                self.move_sensor(view)

    def start(self, initial_view=None):
        self.move_sensor(initial_view)

        # self.init_camera_pose(initial_view)
        # while self.step < self.planning_budget:
        #     next_view = self.plan_next_view()
        #     self.move_sensor(next_view)

        # self.record_experiment()
        # print("------ complete mission ------\n")
        # rospy.signal_shutdown("shut down ros node")

    def move_sensor(self, view):
        pose = utils.view_to_pose_with_target_point(view)
        pub_quaternion = utils.rotation_2_quaternion(pose[:3, :3])
        pub_position = pose[:3, -1]
        pub_pose = [*pub_position, *pub_quaternion]
        print(
            f"------flying to given pose and take measurement No.{self.step + 1} ------\n"
        )
        # current_pose = [*current_position, *current_orientation]
        pub_time = time.time()
        self.simulator_bridge.slow_move_uav_in_rotations(pub_pose)
        while not self.simulator_bridge.check_if_uav_arrive(pub_pose): #[cyw]: check if uav reach
            # print("flying")
            self.simulator_bridge.slow_move_uav_in_rotations(pub_pose)
            time.sleep(0.1)
        self.simulator_bridge.slow_move_uav_in_rotations(pub_pose)
        #[cyw]: check again
        while not self.simulator_bridge.check_if_uav_arrive(pub_pose): #[cyw]: check if uav reach
            pass
        time.sleep(2.5)

        os.makedirs(self.record_path, exist_ok = True)
        csv_file = (f"{self.record_path}/target_uav_pose.csv")
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp','x', 'y', 'z', 'qx', 'qy', 'qz','qw'])
            pose_for_csv = [pub_time, *pub_pose]
            writer.writerow(pose_for_csv)
        self.step += 1

    def plan_next_view(self):

        raise NotImplementedError("plan_next_view method is not implemented")

    def record_experiment(self): #[cyw]:record 
        print("------ record experiment data ------\n")

        os.makedirs(self.record_path, exist_ok=True)
        images_path = os.path.join(self.record_path, "images")
        os.mkdir(images_path)
        depths_path = os.path.join(self.record_path, "depths")
        os.mkdir(depths_path)

        for i, rgb in enumerate(self.rgb_measurements):
            imageio.imwrite(
                f"{images_path}/{i+1:04d}.png", (rgb * 255).astype(np.uint8)
            )

        if len(self.depth_measurements) > 0:
            for i, depth in enumerate(self.depth_measurements):
                with open(f"{depths_path}/depth_{i+1:04d}.npy", "wb") as f:
                    depth_array = np.array(depth, dtype=np.float32)
                    np.save(f, depth_array)

        with open(f"{self.record_path}/trajectory.npy", "wb") as f:
            np.save(f, self.trajectory)

        with open(f"{self.record_path}/camera_info.yaml", "w") as f:
            yaml.safe_dump(self.camera_info, f)

        # record json data required for instant-ngp training
        utils.record_render_data(self.record_path, self.camera_info, self.trajectory)

    def record_step(self, view, pose, rgb, depth):
        self.record_trajectory(view, pose)
        self.record_rgb_measurement(rgb)
        if depth is not None:
            self.record_depth_measurement(depth)

    def record_rgb_measurement(self, rgb):
        rgb = np.clip(rgb, a_min=0, a_max=255)
        rgb = rgb / 255
        self.rgb_measurements[self.step] = rgb

    def record_depth_measurement(self, depth):
        self.depth_measurements[self.step] = depth

    def record_trajectory(self, view, pose):
        self.view_trajectory[self.step] = view
        self.trajectory[self.step] = pose

    #[cyw]
    def get_record_path(self):
        return self.record_path
    
    def store_test_set(self):
        transform_json_file = (f"{self.record_path}/transforms_test.json")
        if os.path.isfile(transform_json_file):
            # Read the existing data
            with open(transform_json_file, 'r') as f:
                record_dict = json.load(f)
        else:
            record_dict = utils.get_camera_json(self.camera_info)
            with open(transform_json_file, "w") as f:
                json.dump(record_dict, f, indent=4)


    def store_train_set(self):
        print("------ record experiment data ------\n")
        # time.sleep(0.5)

        os.makedirs(self.record_path, exist_ok=True)

        train_path = os.path.join(self.record_path, "train")
        os.makedirs(train_path, exist_ok=True)
        depths_path = os.path.join(self.record_path, "depths")
        os.makedirs(depths_path, exist_ok=True)
        
        rgb, depth, captured_pose = self.simulator_bridge.get_image()
        # with open(f"{self.record_path}/trajectory.npy", "wb") as f:
        #     np.save(f, self.trajectory)

        #[cyw]: save camera_info yaml file
        with open(f"{self.record_path}/camera_info.yaml", "w") as f:
            yaml.safe_dump(self.camera_info, f)
        
        #[cyw]: save the transform json file

        transform_json_file = (f"{self.record_path}/transforms_train.json")
        if os.path.isfile(transform_json_file):
            # Read the existing data
            with open(transform_json_file, 'r') as f:
                record_dict = json.load(f)
        else:
            record_dict = utils.get_camera_json(self.camera_info)
            with open(transform_json_file, "w") as f:
                json.dump(record_dict, f, indent=4)
        i = len(record_dict["frames"])  
                
        #[cyw]: change i to the index of image
        train_image_file = (f"./train/r_{i:04d}")
        imageio.imwrite(f"{train_path}/r_{i:04d}.png", rgb)


        if depth != None:
            with open(f"{depths_path}/depth_{i:04d}.npy", "wb") as f:
                depth_array = np.array(depth, dtype=np.float32)
                np.save(f, depth_array)  
        # transformation = np.array(
        # [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        # )  # transform gazebo coordinate to opengl format
        # transform = utils.quaternion_to_rotation_matrix(captured_pose)
        # print(transform)
        # opengltransform = transform @ transformation

        transform = utils.quaternion_to_rotation_matrix(captured_pose)
        opengltransform = np.eye(4)
        euler = R.from_matrix(transform[:3, :3]).as_euler('xyz')
        # print(f"origin euler: roll: {euler[0]}, pitch: {euler[1]}, yaw:{euler[2]}")
        new_euler = np.empty(3)
        new_euler[0] = euler[1] + (np.pi/2)
        new_euler[1] = euler[0]
        new_euler[2] = euler[2] - (np.pi/2)
        opengl_rotation =  R.from_euler('xyz', [new_euler[0] , new_euler[1], new_euler[2]]).as_matrix()
        opengltransform[:3, -1] = transform[:3, -1]
        opengltransform[:3, :3] = opengl_rotation

        data_frame = {
            "file_path": train_image_file,
            "transform_matrix": opengltransform.tolist(),
        }
        # #[cyw]:chage json file function for checking image i/o
    
        record_dict["frames"].append(data_frame)
        with open(transform_json_file, "w") as f:
            json.dump(record_dict, f, indent=4)
    
    def sampling_view(self, sampling_method, num):
        if sampling_method  == "random":
            view_list = np.empty((num, 3))
            for i in range(num):
                view_list[i] = utils.uniform_sampling(3, 10, self.phi_min)
        elif sampling_method  == "circular":
            view_list = utils.sphere_sampling(longtitude_range = 16, latitude_range = 4,
                                                   radius_start = self.radius_start, radius_end =self.radius_end)
        return view_list
    
    def sort_view(self, view_list):
        sorted_indices = np.lexsort((view_list[:,1], view_list[:,0], view_list[:,2]))
        view_list = view_list[sorted_indices]
        return view_list
    
    def record_running_time(self, record_type, index, exec_times):
        if record_type == "uncertainty":
            time_csv = f"{self.record_path}/uncertainty_time.csv"
        elif record_type == "planning":
            time_csv = f"{self.record_path}/planning_time.csv"
        else:
            time_csv = f"{self.record_path}/not_defined.csv"
        
        file_exists = os.path.isfile(time_csv)
        if not file_exists:
            with open(time_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['idxs', 'times'])
                writer.writerow([index, exec_times])  
        else:
            with open(time_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([index, exec_times])