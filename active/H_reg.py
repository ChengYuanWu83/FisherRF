import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Tuple
from copy import deepcopy
import random
from gaussian_renderer import render, network_gui, modified_render
from scene import Scene


class HRegSelector(torch.nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        self.seed = args.seed
        self.reg_lambda = args.reg_lambda
        self.I_test: bool = args.I_test
        self.I_acq_reg: bool = args.I_acq_reg

        name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
        self.filter_out_idx: List[str] = [name2idx[k] for k in args.filter_out_grad]

    
    def nbvs(self, gaussians, scene: Scene, num_views, pipe, background, exit_func) -> List[int]:
        candidate_views = list(deepcopy(scene.get_candidate_set()))
        # print(f"candidate_views {candidate_views}")
        viewpoint_cams = scene.getTrainCameras().copy()

        if self.I_test == True:
            viewpoint_cams = scene.getTestCameras()

        params = gaussians.capture()[1:7]
        # [cyw]:check what is param
        # for i, element in enumerate(params):
        #     if isinstance(element, torch.Tensor):
        #         print(f"Size of tensor {i}: {element.size()}")
        #     else:
        #         print(f"Element {i} is not a tensor")
        params = [p for i, p in enumerate(params) if i not in self.filter_out_idx]

        # off load to cpu to avoid oom with greedy algo
        device = params[0].device if num_views == 1 else "cpu"
        # device = "cpu" # we have to load to cpu because of inflation

        H_train = torch.zeros(sum(p.numel() for p in params), device=params[0].device, dtype=params[0].dtype)

        candidate_cameras = scene.getCandidateCameras()
        print(f"candidate_cameras: {candidate_cameras[0]}")
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
            return self.select_single_view(H_train, candidate_cameras, candidate_views, gaussians, pipe, background, params, exit_func)
        
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

        for _ in range(num_views):
            I_train = torch.reciprocal(H_train + self.reg_lambda)
            if self.I_acq_reg:
                acq_scores = np.array([torch.sum((cur_H + self.I_acq_reg) * I_train).item() for cur_H in H_candidates])
            else:
                acq_scores = np.array([torch.sum(cur_H * I_train).item() for cur_H in H_candidates])
            selected_idx = acq_scores.argmax()
            selected_idxs.append(candidate_views.pop(selected_idx))

            H_train += H_candidates.pop(selected_idx)

        return selected_idxs

    
    def forward(self, x):
        return x
    
    
    def select_single_view(self, I_train, candidate_cameras, candidate_views, gaussians, pipe, background, params, exit_func, num_views=1):
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

            if self.I_acq_reg:
                I_acq += self.reg_lambda

            gaussians.optimizer.zero_grad(set_to_none = True) 
            acq_scores[idx] += torch.sum(I_acq * I_train).item()
        
        print(f"acq_scores: {acq_scores.tolist()}")
        if self.I_test == True:
            acq_scores *= -1

        _, indices = torch.sort(acq_scores, descending=True)
        # [cyw]: selected views 
        # test loop: 
        #for i in indices[-1:].tolist(): # pick least score
        for i in indices[:num_views].tolist(): # pick highest score
            selected_idxs = [candidate_views[i]]
            # [cyw]: print selected view index and its score
            print(f"sorted acq_scores: {_.tolist()}")
            print(f"indices: {indices.tolist()}")
            print(f"selected_idxs: {selected_idxs}")
            print(f"acq_scores: {acq_scores[i].tolist()}")
        return selected_idxs
