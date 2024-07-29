import numpy as np
from planner.utils import sphere_sampling, view_to_pose_with_target_point
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

radius_start =3
radius_end =5 
radius = np.random.uniform(low=radius_start, high=radius_end)
print(radius)