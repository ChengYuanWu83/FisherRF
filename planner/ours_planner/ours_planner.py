import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Tuple
from copy import deepcopy
import random
from gaussian_renderer import render, network_gui, modified_render
from scene import Scene
from planner.planner import Planner
from planner.utils import view_to_pose_with_target_point, random_view, uniform_sampling, view_to_cam, view_to_xyz,rotation_2_quaternion
import pandas as pd
import time
import threading
import sys
import heapq

class OursPlanner(Planner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_candidates = cfg["num_candidates"]
        self.view_change = cfg["view_change"]
        self.planning_type = cfg["planning_type"] #local or global
        self.planning_method = cfg["planning_method"]
        self.time_constraint = cfg["time_constraint"]
        self.scheduling_window = cfg["scheduling_window"]

        self.timeout_flag = False
        self.view_index = 0
        self.candidate_view_list = None
        self.first_update = 1

        #self.seed = args.seed
        self.reg_lambda = 1e-6
        filter_out_grad = ["rotation"]
        # self.I_test: bool = args.I_test
        # self.I_acq_reg: bool = args.I_acq_reg

        name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
        self.filter_out_idx: List[str] = [name2idx[k] for k in filter_out_grad]

    def nbvs(self, gaussians, scene: Scene, num_views, pipe, background, candidate_cams, exit_func, required_uncer=0) -> List[int]:
#        candidate_views = list(deepcopy(scene.get_candidate_set()))
        # [cyw]: candidate_views is a list of image index
        viewpoint_cams = scene.getTrainCameras().copy()

        # if self.I_test == True:
        #     viewpoint_cams = scene.getTestCameras()

        params = gaussians.capture()[1:7]
        params = [p for i, p in enumerate(params) if i not in self.filter_out_idx]

        # off load to cpu to avoid oom with greedy algo
        device = params[0].device if num_views == 1 else "cpu"
        # device = "cpu" # we have to load to cpu because of inflation

        H_train = torch.zeros(sum(p.numel() for p in params), device=params[0].device, dtype=params[0].dtype)

        candidate_cameras = candidate_cams
        # Run heesian on training set
        for cam in tqdm(viewpoint_cams, desc="Calculating diagonal Hessian on training views"):
            if exit_func():
                raise RuntimeError("csm should exit early")

            render_pkg = modified_render(cam, gaussians, pipe, background)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))

            cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])

            H_train += cur_H

            gaussians.optimizer.zero_grad(set_to_none = True) 

        H_train = H_train.to(device)
        # Run heesian on candidates set if select 1 view
        if required_uncer == 1: #[cyw]: return uncertainty and correspond viewport
            return self.uncer_estimate(H_train, candidate_cameras, gaussians, pipe, background, params, exit_func)
        if num_views == 1:
            return self.select_single_view(H_train, candidate_cameras, gaussians, pipe, background, params, exit_func)
        
        # Run heesian on candidates set if select > 1 views
        H_candidates = []
        for idx, cam in enumerate(tqdm(candidate_cameras, desc="Calculating diagonal Hessian on candidate views")):
            if exit_func():
                raise RuntimeError("csm should exit early")

            render_pkg = modified_render(cam, gaussians, pipe, background)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))

            cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])

            H_candidates.append(cur_H.to(device))

            gaussians.optimizer.zero_grad(set_to_none = True) 
        
        selected_idxs = []
        
        # for _ in range(num_views): #[cyw]: this loop is for selecting a batch of view in each time
        #     I_train = torch.reciprocal(H_train + self.reg_lambda)
        #     # if self.I_acq_reg:
        #     #     acq_scores = np.array([torch.sum((cur_H + self.I_acq_reg) * I_train).item() for cur_H in H_candidates])
        #     # else:
        #     #    acq_scores = np.array([torch.sum(cur_H * I_train).item() for cur_H in H_candidates])
        #     acq_scores = np.array([torch.sum(cur_H * I_train).item() for cur_H in H_candidates])

        #     selected_idx = acq_scores.argmax()
        #     selected_idxs.append(candidate_views.pop(selected_idx))

        #     H_train += H_candidates.pop(selected_idx)

        return selected_idxs

    
    def forward(self, x):
        return x
    
    
    def select_single_view(self, I_train, candidate_cameras, gaussians, pipe, background, params, exit_func, num_views=1):
        """
        A memory effcient way when doing single view selection
        """
        I_train = torch.reciprocal(I_train + self.reg_lambda)
        acq_scores = torch.zeros(len(candidate_cameras))

        for idx, cam in enumerate(tqdm(candidate_cameras, desc="Calculating diagonal Hessian on candidate views")):
            if exit_func():
                raise RuntimeError("csm should exit early")

            render_pkg = modified_render(cam, gaussians, pipe, background)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))

            cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])

            I_acq = cur_H

            # if self.I_acq_reg:
            #     I_acq += self.reg_lambda

            gaussians.optimizer.zero_grad(set_to_none = True) 
            acq_scores[idx] += torch.sum(I_acq * I_train).item()
        
        print(f"acq_scores: {acq_scores.tolist()}")
        # if self.I_test == True:
        #     acq_scores *= -1

        _, indices = torch.sort(acq_scores, descending=True)
        # [cyw]: selected views 
        # test loop: 
        #for i in indices[-1:].tolist(): # pick least score
        for i in indices[:num_views].tolist(): # pick highest score
            selected_idxs = i
            # [cyw]: print selected view index and its score
            print(f"sorted acq_scores: {_.tolist()}")
            print(f"indices: {indices.tolist()}")
            print(f"selected_idxs: {selected_idxs}")
            print(f"acq_scores: {acq_scores[i].tolist()}")
        return selected_idxs
    
    def plan_next_view(self, gaussians, scene: Scene, num_views, pipe, background, exit_func):
        view_list = np.empty((self.num_candidates, 3))
        
        current_pose = []
        cp = self.simulator_bridge.current_pose
        current_pose.append(cp.transform.translation.x)
        current_pose.append(cp.transform.translation.y)
        current_pose.append(cp.transform.translation.z)
        current_pose = np.array(current_pose)
        if self.planning_type == "local":
            for i in range(self.num_candidates):
                view_list[i] = random_view(
                    current_pose, #[cyw]: change planner pose to ros pose, oringinal: self.current_pose[:3, 3]
                    self.radius_start, self.radius_end,
                    self.phi_min,
                    min_view_change=0.2,
                    max_view_change=self.view_change,
                )
            #print(f"view_list: {view_list}") #[cyw]: phi, theta
        elif self.planning_type == "global":
            for i in range(self.num_candidates):
                view_list[i] = uniform_sampling(self.radius_start, self.radius_end, self.phi_min)

        candidate_cams = []
        # [cyw]: transform random_view(phi, theta) to cam in order to get the novel view
        for view in view_list:
            candidate_cams.append(view_to_cam(view, self.camera_info))

        # for cam in candidate_cams:

        #     print(cam.world_view_transform,
        #           cam.projection_matrix,
        #           cam.full_proj_transform,
        #           cam.camera_center
        #           )


        #nbv_index = np.random.choice(len(view_list))
        nbv_index = self.nbvs(gaussians, scene, num_views, pipe, background, candidate_cams, exit_func)

        nbv = view_list[nbv_index]
        print(f"ours planner select view: {nbv}")
        return nbv
    
    def view_to_camera_info(view):
        return 1
    
    def uncer_estimate(self, I_train, candidate_cameras, gaussians, pipe, background, params, exit_func, num_views=1):
        """
        A memory effcient way when doing single view selection
        """
        I_train = torch.reciprocal(I_train + self.reg_lambda)
        acq_scores = torch.zeros(len(candidate_cameras))

        for idx, cam in enumerate(tqdm(candidate_cameras, desc="Calculating diagonal Hessian on candidate views")):
            if exit_func():
                raise RuntimeError("csm should exit early")

            render_pkg = modified_render(cam, gaussians, pipe, background)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))

            cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])

            I_acq = cur_H

            # if self.I_acq_reg:
            #     I_acq += self.reg_lambda

            gaussians.optimizer.zero_grad(set_to_none = True) 
            acq_scores[idx] += torch.sum(I_acq * I_train).item()
        
        # if self.I_test == True:
        #     acq_scores *= -1

        return  acq_scores.tolist()

    def first_plan(self, sampling_method, num):
        view_list = self.sampling_view(sampling_method, num, 0)
        if sampling_method  == "circular":
            self.candidate_view_list = view_list
            view_list = self.candidate_view_list[self.view_index:self.view_index+num]
            self.view_index += num
        # view_list = self.sort_view(view_list)
        self.first_update = 1
        print(view_list)
        return view_list

    def plan_path(self, gaussians, scene: Scene, num_views, pipe, background, exit_func):
        required_uncer = 1

        # time_budget = experiment_params["time_budget"]
        sampling_method = self.sampling_method
        sampling_num = self.sampling_num
        planning_method = self.planning_method
        scheduling_window = self.scheduling_window

        uncer_start = time.time()
        if sampling_method  == "random":
            view_list = self.sampling_view(sampling_method, sampling_num)
        elif sampling_method  == "circular":
            if self.planning_time == 0:
                self.candidate_view_list = self.sampling_view(sampling_method, sampling_num)
            
            base = self.planning_time * 80
            if self.planning_time > 7:
                base = ((self.planning_time % 8) - 1) * 80 + 40
            view_list = self.candidate_view_list[base:base+sampling_num]
            # self.view_index += sampling_num
        print(view_list)
        

        candidate_cams = []
        view_list_stored = []
        # [cyw]: transform random_view(phi, theta) to cam in order to get the novel view
        for view in view_list:
            candidate_cams.append(view_to_cam(view, self.camera_info))
            pose = view_to_pose_with_target_point(view)
            rotation = rotation_2_quaternion(pose[:3, :3])
            translation = pose[:3, -1]
            view_list_stored.append([*translation, *rotation])

        df = pd.DataFrame(view_list_stored)
        sampling__list_header = ['x', 'y', 'z', 'qx', 'qy', 'qz','qw']
        df.to_csv(f'{self.record_path}/sampling_list_{self.planning_time}.csv', index=False, header=sampling__list_header)


        uncertainty = self.nbvs(gaussians, scene, num_views, pipe, background, candidate_cams, exit_func, required_uncer)
        uncertainty = np.nan_to_num(uncertainty, nan=0.0)
        uncertainty =  np.insert(uncertainty, 0, 0)
        uncer_end = time.time()
        self.record_running_time("uncertainty", self.planning_time, uncer_end - uncer_start)

        xyz_list = np.zeros_like(view_list)
        for i, view in enumerate(view_list):
            xyz_list[i] = view_to_xyz(view)
        
        # [cyw]: adding current pose to list
        current_pose = []
        cp = self.simulator_bridge.current_pose
        current_pose.append(cp.transform.translation.x)
        current_pose.append(cp.transform.translation.y)
        current_pose.append(cp.transform.translation.z)
        current_pose = np.array(current_pose)
        xyz_list = np.insert(xyz_list, 0, current_pose, axis=0)

        n = len(xyz_list)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    distance = 0
                else:
                    distance = np.linalg.norm(xyz_list[i] - xyz_list[j])
                    distance += 2.5 #[cyw]: uav wait for stable
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance


        df = pd.DataFrame(uncertainty)
        df.to_csv(f'{self.record_path}/uncertainty_{self.planning_time}.csv', index=False, header=True)
        
        df = pd.DataFrame(np.array(distance_matrix))
        df.to_csv(f'{self.record_path}/time_matrix_{self.planning_time}.csv', index=False, header=True)   
        


        planning_start = time.time()
        if planning_method == "dp": 
            print("start DP planning")
            # max_util, best_path = self.find_max_utility_with_dp(uncertainty, distance_matrix, training_time_limit)
            max_util, best_path = find_max_utility_with_test(uncertainty, distance_matrix, scheduling_window, self.time_constraint)

        elif planning_method == "astar":
            print("start A* planning")
            max_util, best_path = max_utility_with_a_star(uncertainty, distance_matrix, scheduling_window)
        else:
            print("unknown planning method, use random")
            return view_list
        
        best_path = np.array(best_path[1:]) - 1
        planning_end = time.time()
        self.record_running_time("planning", self.planning_time, planning_end - planning_start)

        print(f"plan the path {best_path} have max utilitiy {max_util}")
        if len(best_path) == 0:
            best_path = [np.argmax(uncertainty)]
        traj = view_list[best_path]
        self.planning_time +=1

        return traj
    
    def need_to_update(self, time_budget, training_time, scheduling_window, initial_training_time, view_list, step):

        # if self.first_update == 1 and loss < 0.065 and training_time > 1.0: #update from loss
        #     self.first_update = 0 
        #     print("first_update")
        #     return True
        if self.first_update == 1 and training_time > initial_training_time: #update from time
            self.first_update = 0 
            print("first_update")
            return True
        
        if view_list is None:
            # print(training_time)
            return False

        if time_budget > 0 and (training_time > scheduling_window or step == len(view_list) - 1):
            return True
        
        return False

    # DP algo
    def max_utility_with_dp(self, i, remaining_time, utility, time_matrix, visited, memo):
        # global timeout_flag

        # 如果計時器超時，則返回0和空路徑
        if self.timeout_flag:
            return 0, []

        # if we run out the time
        if remaining_time <= 0:
            return 0, []

        # 如果計算過了，就直接返回結果
        if (i, remaining_time, tuple(visited)) in memo:
            return memo[(i, remaining_time, tuple(visited))]

        max_util = 0
        best_path = []

        for j in range(len(utility)):
            if not visited[j] and remaining_time >= time_matrix[i][j]:
                visited[j] = True
                util, sub_path = self.max_utility_with_dp(j, remaining_time - time_matrix[i][j], utility, time_matrix, visited, memo)
                visited[j] = False
                util += utility[j]
                if util > max_util:
                    max_util = util
                    best_path = [j] + sub_path

        memo[(i, remaining_time, tuple(visited))] = (max_util, best_path)
        return max_util, best_path

    def find_max_utility_with_dp(self, utility, time_matrix, total_time):
        # global timeout_flag

        def set_timeout():
            # global timeout_flag
            self.timeout_flag = True
            sys.exit("計算時間超過限制")

        # 設置一個計時器，1秒後設置超時旗標
        timer = threading.Timer(1, set_timeout)
        timer.start()

        memo = {}
        visited = [False] * len(utility)
        visited[0] = True  # 從點0開始

        try:
            max_util, best_path = self.max_utility_with_dp(0, total_time, utility, time_matrix, visited, memo)
            best_path = [0] + best_path  # 加上起始點0
            max_util += utility[0]  # 加上起始點0的效用
        except SystemExit as e:
            print(e)
            max_util, best_path = 0, []

        # 停止計時器
        timer.cancel()

        return max_util, best_path


# A* algo
def max_utility_with_a_star(utilities, time_matrix, limit_time):
    n = len(utilities)

    # 初始化佇列
    queue = [(0, 0, 0, [0])]  # (current utility g(i), current time, current point, path)

    max_utility = 0
    best_path = []

    while queue:
        current_utility, current_time, current_point, path = queue.pop(0)

        # if current_time > limit_time:
        #     continue

        if current_utility > max_utility:
            max_utility = current_utility
            best_path = path
            if len(best_path) == n:
                print("已經飛過所有的點了")
                return max_utility, best_path

        max_f_value = 0

        for next_point in range(n):
            if next_point not in path:
                time_to_next = current_time + time_matrix[current_point][next_point]
                if time_to_next <= limit_time:
                    new_utility = current_utility + utilities[next_point]
                    remaining_time = limit_time - time_to_next
                    h_value = estimate_heuristic(utilities, next_point, path, remaining_time, time_matrix)
                    f_value = new_utility + h_value
                    if f_value > max_f_value:
                        max_f_value = f_value
                        new_item = (new_utility, time_to_next, next_point, path + [next_point])
                        if len(queue) > 0:
                            queue[0] = new_item
                        else:
                            queue.append(new_item)

    return max_utility, best_path

def estimate_heuristic(utility, current_point, path, remaining_time, time_matrice):
    # 簡單估算剩餘時間內可以獲得的最大 utility
    if remaining_time <= 0:
        return 0

    max_utility = 0
    for next_point in range(len(utility)):
        if next_point not in path and time_matrice[current_point][next_point] != 0:
            if utility[next_point] / time_matrice[current_point][next_point] > max_utility:
                max_utility = utility[next_point] / time_matrice[current_point][next_point]
    return max_utility * remaining_time

# DP2
def find_max_utility_with_test(utilities, time_matrix, t_max, time_constraint=1.0):
    start_time = time.time()
    n = len(utilities)
    dp = {}
    # Max heap to prioritize paths with higher utility
    heap = [(-utilities[0], 0, 0, [0])]
    max_utility = 0
    best_path = []

    while heap:
        
        if time.time() - start_time > time_constraint:  # 檢查是否超過1秒
            break

        current_utility, current_time, current_point, path = heapq.heappop(heap)
        current_utility = -current_utility
        
        # Update max utility and best path
        if current_utility > max_utility:
            max_utility = current_utility
            best_path = path
        
        for next_point in range(n):
            if next_point not in path:
                next_time = current_time + time_matrix[current_point][next_point]
                if next_time <= t_max:
                    next_utility = current_utility + utilities[next_point]
                    next_path = path + [next_point]
                    state = (next_point, next_time)
                    
                    # If this state is better, push it into the heap
                    if state not in dp or dp[state] < next_utility:
                        dp[state] = next_utility
                        heapq.heappush(heap, (-next_utility, next_time, next_point, next_path))
    
    return max_utility, best_path