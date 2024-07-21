import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Tuple
from copy import deepcopy
import random
from gaussian_renderer import render, network_gui, modified_render
from scene import Scene
from planner.planner import Planner
from planner.utils import view_to_pose_batch, random_view, uniform_sampling, view_to_cam

class FisherPlanner(Planner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_candidates = cfg["num_candidates"]
        self.view_change = cfg["view_change"]
        self.planning_type = cfg["planning_type"]

        #self.seed = args.seed
        self.reg_lambda = 1e-6
        filter_out_grad = ["rotation"]
        # self.I_test: bool = args.I_test
        # self.I_acq_reg: bool = args.I_acq_reg

        name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
        self.filter_out_idx: List[str] = [name2idx[k] for k in filter_out_grad]

    def nbvs(self, gaussians, scene: Scene, num_views, pipe, background, candidate_cams, exit_func) -> List[int]:
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
        view_list = np.empty((self.num_candidates, 2))
        
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
                    self.radius,
                    self.phi_min,
                    min_view_change=0.2,
                    max_view_change=self.view_change,
                )
            #print(f"view_list: {view_list}") #[cyw]: phi, theta
        elif self.planning_type == "global":
            for i in range(self.num_candidates):
                view_list[i] = uniform_sampling(self.radius, self.phi_min)

        candidate_cams = []
        # [cyw]: transform random_view(phi, theta) to cam in order to get the novel view
        for view in view_list:
            candidate_cams.append(view_to_cam(view, self.radius, self.camera_info))

        # for cam in candidate_cams:

        #     print(cam.world_view_transform,
        #           cam.projection_matrix,
        #           cam.full_proj_transform,
        #           cam.camera_center
        #           )


        #nbv_index = np.random.choice(len(view_list))
        nbv_index = self.nbvs(gaussians, scene, num_views, pipe, background, candidate_cams, exit_func)

        nbv = view_list[nbv_index]
        return nbv
    
    def view_to_camera_info(view):
        return 1