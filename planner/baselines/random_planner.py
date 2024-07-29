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
        self.radius_start = 3
        self.radius_end = 5

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