import numpy as np 
from planner.utils import uniform_sampling, xyz_to_view
import pandas as pd
import os
import csv

# views_num = 20
# scheduling_num = 3
# radius_start = 4
# radius_end = 10
# phi_min = 0.2617
# main_size = int(views_num) * int(scheduling_num)
# main_view_list = np.empty((main_size, 3))
# for i in range(main_size):
#     seed = i
#     main_view_list[i] = uniform_sampling(radius_start, radius_end, phi_min, seed)

# print(main_view_list)
x =11
print(((x % 10) - 1) * 80 + 40)