from scipy.spatial.transform import Rotation as R
from planner import utils
import numpy as np

if __name__ == "__main__":
    source = [-2,9,1]


    # roll = 0
    # pitch = 0
    # yaw = -90 
    pose = utils.points_to_pose(source)
    #original transform
    old_transformation = np.array(
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
    )
    # rotate the blender camera to face the direction that same as the gazebo camera 
    rotation_blender_z = R.from_euler('z', -90 , degrees=True).as_matrix()
    rotation_blender_y = R.from_euler('y', -90 , degrees=True).as_matrix()
    rotation_blender_x = R.from_euler('x', 90 , degrees=True).as_matrix()
    # rotation = np.eye(3)
    rotation = pose[:3, :3]
    euler = R.from_matrix(rotation).as_euler('xyz')
    print(f"origin euler: roll: {euler[0]}, pitch: {euler[1]}, yaw:{euler[2]}")
    new_euler = np.empty(3)
    new_euler[0] = euler[1] + (np.pi/2)
    new_euler[1] = euler[0]
    new_euler[2] = euler[2] - (np.pi/2)
    print(f"new_euler:roll: {new_euler[0]}, pitch: {new_euler[1]}, yaw:{new_euler[2]}")
    # final_rotation = rotation @ rotation_blender_z @ rotation_blender_x
    # # final_rotation = rotation @ old_transformation


    # euler = R.from_matrix(final_rotation).as_euler('xyz',degrees=True)
    # quat = R.from_matrix(final_rotation).as_quat()
    # print(f"quat: {quat}")
    # print(f"euler: {euler}")