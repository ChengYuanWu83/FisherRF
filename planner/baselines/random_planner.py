from planner.planner import Planner
from planner.utils import random_view, uniform_sampling, sphere_sampling
import numpy as np


class RandomPlanner(Planner):
    def __init__(self, cfg):
        super().__init__(cfg)
        print("initial ")
        self.num_candidates = cfg["num_candidates"]
        self.view_change = cfg["view_change"]
        self.planning_type = cfg["planning_type"]


        self.candidate_view_list = sphere_sampling(longtitude_range = 16, latitude_range = 4,
                                                   radius_start = self.radius_start, radius_end =self.radius_end) 

    def plan_next_view(self):
        # view_list = np.empty((self.num_candidates, 2))

        view_list = self.candidate_view_list
        # print(f"view_list_len: {len(view_list)}")
        nbv_index = np.random.choice(len(view_list))
        nbv = view_list[nbv_index]
        print(f"randomly select view: {nbv}")
        view_list = np.array([item for item in view_list if not np.array_equal(item, nbv)])
        self.candidate_view_list = view_list
        return nbv
    
    def del_init_view(self,init_view):
        view_list = self.candidate_view_list
        view_list = np.array([item for item in view_list if not np.array_equal(item, init_view)])
        self.candidate_view_list = view_list
    
    def first_plan(self, sampling_method, sampling_num):
        return self.plan_path(sampling_method, sampling_num)

    def plan_path(self, sampling_method, sampling_num):
        # view_list = np.empty((self.num_candidates, 2))
        # sampling_method = "random"
        # num = 50
        view_list = self.sampling_view(sampling_method, sampling_num)
        # view_list = self.sort_view(view_list)
        return view_list
    
    def need_to_update(self, time_budget, training_time, training_time_limit, loss, view_list, step):
        if view_list is None:
            return True
        
        if time_budget > 0 and step == len(view_list) - 1:
            return True
        return False  
