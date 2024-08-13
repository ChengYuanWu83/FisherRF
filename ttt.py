import numpy as np 
from planner.planner import Planner
import pandas as pd
import os
import csv



# x = 123
# y =888

# algo_time_csv = f"./algo_time.csv"
# # algo_file_exists = os.path.isfile(algo_time_csv)
# # if not algo_file_exists:
# with open(algo_time_csv, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['iterations', 'times'])
#     writer.writerow([x, y])


x = np.array([1231230,789,978,78,55,4])
i= [np.argmin(x)]

y = x[1:]
if len(y) == 0:
    print("123")
print(i)
print(y)
