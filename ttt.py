import numpy as np
from planner.utils import sphere_sampling, view_to_pose_with_target_point
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def slerp(q1, q2, num_intervals):
    # Create Rotation objects from quaternions
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    print(r1.as_euler('xyz', degrees=True),r2.as_euler('xyz', degrees=True))
    key_times = [0, 1]
    key_rots = R.from_quat([r1.as_quat(), r2.as_quat()])
    # Generate the interpolation values
    
    
    # Perform the slerp interpolation
    slerped_rots = Slerp(key_times, key_rots)
    
    # # Extract the quaternions from the interpolated rotations
    # slerped_quats = slerped_rots.as_quat()
    # print(slerped_rots)
    return slerped_rots

q1 = [1, 0, 0, 0]  # Example source quaternion
q2 = [0, 1, 0, 0]  # Example target quaternion

# Get 10 intervals
num_intervals = 10
times = np.linspace(0, 1, num_intervals + 1)
interpolated_quaternions = slerp(q1, q2, num_intervals)
x = interpolated_quaternions(times)
print(x[0].as_quat())
# print(interpolated_quaternions(times).as_euler('xyz', degrees=True))


# pose=[1,2,3,4,5,6,7]
# print(pose[3:])
# print(pose[:3])