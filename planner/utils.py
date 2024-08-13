from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
import json

# from rembg import remove
import os
import imageio
from scene.cameras import FakeCam
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped
import time

def get_roi_mask(rgb):
    """binary mask for ROIs using color thresholding"""
    hsv = cv2.cvtColor(np.array(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([20, 255, 255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([160, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask0 + mask1
    mask = mask + 1e-5

    return mask


def get_black_mask(rgb):
    """binary mask for ROIs using color thresholding"""
    lower_black = np.array([250, 250, 250])
    upper_black = np.array([255, 255, 255])
    mask = cv2.inRange(rgb, lower_black, upper_black)

    return mask


def visualize_uncertainty(uncertainty):
    variance = np.exp(uncertainty)


def rotation_2_quaternion(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    return r.as_quat()

def xyz_to_view_cylinder(xyz, radius): #[cyw]: cylinder look inside
    phi = np.arctan2(xyz[2], radius)  # phi from 0 to 0.5*pi, vertical
    theta = np.arctan2(xyz[1], xyz[0]) % (2 * np.pi)  # theta from 0 to 2*pi, horizontal

    return [phi, theta]




def xyz_to_view(xyz, radius): #half sphere look inside
    phi = np.arcsin(xyz[2] / radius)  # phi from 0 to 0.5*pi, vertical
    theta = np.arctan2(xyz[1], xyz[0]) % (2 * np.pi)  # theta from 0 to 2*pi, horizontal

    return [phi, theta, radius]

def view_to_xyz(view):
    phi, theta, radius = view
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return [x, y, z]


def view_to_pose(view, radius):
    phi, theta = view

    # phi should be within [min_phi, 0.5*np.pi)
    if phi >= 0.5 * np.pi:
        phi = np.pi - phi

    pose = np.eye(4)
    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.cos(phi)
    z = radius * np.sin(phi)

    translation = np.array([x, y, z])
    rotation = R.from_euler("ZYX", [theta, np.pi+phi, np.pi]).as_matrix()

    pose[:3, -1] = translation
    pose[:3, :3] = rotation
    return pose

def view_to_pose_with_target_point(view, target=[0, 0, 0]):
    phi, theta, radius= view

    # phi should be within [min_phi, 0.5*np.pi)
    if phi >= 0.5 * np.pi:
        phi = np.pi - phi

    pose = np.eye(4)
    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.cos(phi)
    z = radius * np.sin(phi)

    translation = np.array([x, y, z])

    pose = np.eye(4)

    source = translation
    target = np.array(target)

    direction = target - source
    
    direction = direction / np.linalg.norm(direction)
    # print(f"direction: {direction}")
    yaw = np.arctan2(direction[1], direction[0])
    pitch = np.arcsin(direction[2])
    roll = 0
    # print(f"roll: {roll}, pitch: {pitch}, yaw:{yaw}")
    # rotation = R.from_euler("ZYX", [theta, np.pi+phi, np.pi]).as_matrix()
    # rotation = R.from_euler('ZYX', [yaw , pitch, 0]).as_matrix()
    rotation = R.from_euler('xyz', [roll , pitch, yaw]).as_matrix()
    pose[:3, -1] = translation
    pose[:3, :3] = rotation
    return pose

def points_to_pose(source, target=[0, 0, 0]):

    pose = np.eye(4)


    translation = np.array(source)

    source = translation
    target = np.array(target)

    direction = target - source
    
    direction = direction / np.linalg.norm(direction)
    # print(f"direction: {direction}")
    yaw = np.arctan2(direction[1], direction[0])
    pitch = np.arcsin(direction[2])
    roll = 0
    print(f"roll: {roll}, pitch: {pitch}, yaw:{yaw}")
    # rotation = R.from_euler("ZYX", [theta, np.pi+phi, np.pi]).as_matrix()
    # rotation = R.from_euler('ZYX', [yaw , pitch, 0]).as_matrix()
    rotation = R.from_euler('xyz', [roll , pitch, yaw]).as_matrix()
    pose[:3, -1] = translation
    pose[:3, :3] = rotation
    return pose

def view_to_position_and_rotation(view, radius): #[cyw]
    phi, theta = view

    # phi should be within [min_phi, 0.5*np.pi)
    if phi >= 0.5 * np.pi:
        phi = np.pi - phi

    # pose = np.eye(4)
    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.cos(phi)
    z = radius * np.sin(phi)

    translation = np.array([x, y, z])
    # rotation_init = R.from_euler("XYZ", [-1.57, -3.14, 0]).as_matrix()
    rotation_angle = R.from_euler("ZYZ", [theta, -phi, np.pi]).as_matrix()
    # rotation_angle =  rotation_angle * rotation_init
    rotation = rotation_2_quaternion(rotation_angle)
    # pose[:3, -1] = translation
    # pose[:3, :3] = rotation
    return translation, rotation


def view_to_pose_batch(views, radius):
    num = len(views)
    phi = views[:, 0]
    theta = views[:, 1]

    # phi should be within [min_phi, 0.5*np.pi)
    index = phi >= 0.5 * np.pi
    phi[index] = np.pi - phi[index]

    poses = np.broadcast_to(np.identity(4), (num, 4, 4)).copy()

    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.cos(phi)
    z = radius * np.sin(phi)

    translations = np.stack((x, y, z), axis=-1)

    angles = np.stack((theta, -phi, np.pi * np.ones(num)), axis=-1)
    rotations = R.from_euler("ZYZ", angles).as_matrix()

    poses[:, :3, -1] = translations
    poses[:, :3, :3] = rotations

    return poses


def random_view(current_xyz, radius_start, radius_end, phi_min, min_view_change, max_view_change):
    """
    random scatter view direction changes by given current position and view change range.
    """
    radius = np.random.uniform(low=radius_start, high=radius_end)
    u = current_xyz / np.linalg.norm(current_xyz)

    # pick a random vector:
    r = np.random.multivariate_normal(np.zeros_like(u), np.eye(len(u)))

    # form a vector perpendicular to u:
    uperp = r - r.dot(u) * u
    uperp = uperp / np.linalg.norm(uperp)

    # random view angle change in radian
    random_view_change = np.random.uniform(low=min_view_change, high=max_view_change)
    cosine = np.cos(random_view_change)
    w = cosine * u + np.sqrt(1 - cosine**2 + 1e-8) * uperp
    w = np.linalg.norm(current_xyz) * w / np.linalg.norm(w)

    view = xyz_to_view(w, radius)

    if view[0] < phi_min:
        view[0] = phi_min
    # if view[0] > 0.5: #[cyw]
    #     view[0] = 0.5

    return view


def uniform_sampling(radius_start, radius_end, phi_min):
    """
    uniformly generate unit vector on hemisphere.
    then calculate corresponding view direction targeting coordinate origin.
    """
    n = int(time.time_ns() % 1000000) #need to change
    np.random.seed(n)

    radius = np.random.uniform(low=radius_start, high=radius_end)
    xyz = np.array([0.0, 0.0, 0.0])

    # avoid numerical error
    while np.linalg.norm(xyz) < 0.001:
        xyz[0] = np.random.uniform(low=-1.0, high=1.0)
        xyz[1] = np.random.uniform(low=-1.0, high=1.0)
        xyz[2] = np.random.uniform(low=0.0, high=1.0)

    xyz = radius * xyz / np.linalg.norm(xyz)
    view = xyz_to_view(xyz, radius)

    if view[0] < phi_min:
        view[0] = phi_min
    # if view[0] > 0.5: #[cyw]
    #     view[0] = 0.5
    return view

def sphere_sampling(longtitude_range, latitude_range, radius_start, radius_end):
    # radius_list = np.arange(start=radius_start, stop=radius_end+1, step=1)
    order = []
    start = radius_start
    end = radius_end
    while start <= end:
        order.append(start)
        start += 1
        if start <= end:
            order.append(end)
            end -= 1
    order.reverse()

    view_list = np.empty((latitude_range * longtitude_range *10, 3))

    latitude_interval = 15
    phi_list = np.arange(1, latitude_range + 1) * latitude_interval * (np.pi / 180)    #[cyw]:phi_list
    theta_list = np.empty(longtitude_range)
    for i in range(longtitude_range):
        theta_list[i] = (360/longtitude_range) * i * (np.pi / 180)

    index = 0 
    # for radius in radius_list:
    #     for phi in phi_list:
    #         for theta in theta_list:
    #             view_list[index][0] = phi
    #             view_list[index][1] = theta
    #             view_list[index][2] = radius
    #             index += 1
    # n = int(time.time_ns() % 1000000) #need to change
    # np.random.seed(n)

    # radius = np.random.uniform(low=radius_start, high=radius_end)
    for i in range(10):
        for phi in phi_list:
            for theta in theta_list:
                r = index % len(order)
                radius = order[r]
                view_list[index][0] = phi
                view_list[index][1] = theta
                view_list[index][2] = radius
                index += 1
    # # sorted_indices = np.argsort(view_list[:, 1])
    # view_list = view_list[sorted_indices]
    return view_list

def focal_len_to_fov(focal, resolution):
    """
    calculate FoV based on given focal length adn image resolution

    Args:
        focal: [fx, fy]
        resolution: [W, H]

    Returns:
        FoV: [HFoV, VFoV]

    """
    focal = np.asarray(focal)
    resolution = np.asarray(resolution)

    return 2 * np.arctan(0.5 * resolution / focal)


def mask_out_background(image_path):
    """remove background"""

    rgb = imageio.imread(image_path)
    masked_rgb = remove(rgb)
    # H, W, _ = rgb.shape
    # masked_rgb = np.ones((H, W, 4)) * 255
    # masked_rgb[..., :3] = rgb
    # mask_white = rgb >= np.array([254, 254, 254])
    # mask_white = np.all(mask_white, axis=-1)
    # mask_black = rgb <= np.array([1, 1, 1])
    # mask_black = np.all(mask_black, axis=-1)
    # masked_rgb[mask_white] = [0, 0, 0, 0]
    # masked_rgb[mask_black] = [0, 0, 0, 0]

    return masked_rgb


def record_render_data(path, camera_info, trajectory, use_masked_image=False):
    transformation = np.array(
        [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )  # transform gazebo coordinate to opengl format
    opencv_trajectory = np.empty(trajectory.shape)

    for i, pose in enumerate(trajectory):
        opencv_trajectory[i] = pose @ transformation

    resolution = camera_info["image_resolution"]
    c = camera_info["c"]
    focal = camera_info["focal"]

    fov = focal_len_to_fov(focal, resolution)

    record_dict = {}
    record_dict["camera_angle_x"] = fov[0]
    record_dict["camera_angle_y"] = fov[1]
    record_dict["fl_x"] = focal[0]
    record_dict["fl_y"] = focal[1]
    record_dict["k1"] = 0.000001
    record_dict["k2"] = 0.000001
    record_dict["p1"] = 0.000001
    record_dict["p2"] = 0.000001
    record_dict["cx"] = c[0]
    record_dict["cy"] = c[1]
    record_dict["w"] = resolution[1]
    record_dict["h"] = resolution[0]
    record_dict["frames"] = []
    record_dict["scale"] = 1.0
    record_dict["aabb_scale"] = 2.0

    for i, pose in enumerate(opencv_trajectory):
        image_file = f"images/{i+1:04d}.png"
        image_path = os.path.join(path, image_file)

        if use_masked_image:
            masked_image = mask_out_background(image_path)
            image_file = f"images/masked_{i+1:04d}.png"
            image_path = os.path.join(path, image_file)
            imageio.imwrite(image_path, masked_image)

        data_frame = {
            "file_path": image_file,
            # "sharpness": 30.0,
            "transform_matrix": pose.tolist(),
        }
        record_dict["frames"].append(data_frame)

    with open(f"{path}/transforms.json", "w") as f:
        json.dump(record_dict, f, indent=4)

    # for test only
    # for i, pose in enumerate(opencv_trajectory[50:]):
    #     data_frame = {
    #         "file_path": f"images/{i+51:04d}.jpg",
    #         "sharpness": 30.0,
    #         "transform_matrix": pose.tolist(),
    #     }
    #     record_dict["frames"].append(data_frame)

    # with open(f"{path}/test_transforms.json", "w") as f:
    #     json.dump(record_dict, f, indent=4)


def test():
    view = []
    # for i in range(5):
    #     new_view = uniform_sampling(2, 0.15)
    #     view.append(new_view)
    #     print("view:", new_view)
    current_xyz = [0, 0, 2]
    for i in range(500):
        local = random_view(current_xyz, 2, 0.15, 0.2, 1.05)
        view.append(local)

    xyz_list = view_to_pose_batch(np.array(view), 2)[..., :3, 3]
    print(xyz_list)
    for xyz in xyz_list:
        view = xyz_to_view(xyz, 2)
        print(view)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(xyz_list[..., 0], xyz_list[..., 1], xyz_list[..., 2])
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    ax.set_zlim(0, 2.5)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    plt.show()

#[cyw]: quaternion_to_rotation_matrix might be wrong
def quaternion_to_rotation_matrix(ros_pose):
    """
    Convert a quaternion into a rotation matrix.

    Parameters:
    quaternion (list or tuple): A list or tuple of 4 elements representing the quaternion (qx, qy, qz, qw).

    Returns:
    np.array: A 3x3 rotation matrix.
    """
    if isinstance(ros_pose, TransformStamped):
        translation = ros_pose.transform.translation
        rotation = ros_pose.transform.rotation
        current_position = [translation.x, translation.y, translation.z]
        current_orientation = [rotation.x, rotation.y, rotation.z, rotation.w]
        # print("Current Position (from TransformStamped):", current_position)
        # print("Current Orientation (from TransformStamped):", current_orientation)
    elif isinstance(ros_pose, Pose):
        current_position = [ros_pose.position.x,
                    ros_pose.position.y,
                    ros_pose.position.z]
        current_orientation = [ros_pose.orientation.x,
                    ros_pose.orientation.y,
                    ros_pose.orientation.z,
                    ros_pose.orientation.w]
        
    x = current_position[0]
    y = current_position[1]
    z = current_position[2]
    qx = current_orientation[0]
    qy = current_orientation[1]
    qz = current_orientation[2]
    qw = current_orientation[3]

    # Compute rotation matrix
    # rotation = np.array([
    #     [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
    #     [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
    #     [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    # ])
    rotation = R.from_quat([qx,qy,qz,qw]).as_matrix()
    translation = [x,y,z]

    transform = np.eye(4)
    transform[:3, -1] = translation
    transform[:3, :3] = rotation

    return transform

def get_camera_json(camera_info):
    resolution = camera_info["image_resolution"]
    c = camera_info["c"]
    focal = camera_info["focal"]

    fov = focal_len_to_fov(focal, resolution)

    record_dict = {}
    record_dict["camera_angle_x"] = fov[0]
    record_dict["camera_angle_y"] = fov[1]
    record_dict["fl_x"] = focal[0]
    record_dict["fl_y"] = focal[1]
    record_dict["k1"] = 0.000001
    record_dict["k2"] = 0.000001
    record_dict["p1"] = 0.000001
    record_dict["p2"] = 0.000001
    record_dict["cx"] = c[0]
    record_dict["cy"] = c[1]
    record_dict["w"] = resolution[1]
    record_dict["h"] = resolution[0]
    record_dict["frames"] = []
    record_dict["scale"] = 1.0
    record_dict["aabb_scale"] = 2.0
    return record_dict

def view_to_cam(view,camera_info):
    transform = view_to_pose_with_target_point(view)

    # [cyw]:rotate to opengl transform
    opengltransform = np.eye(4)
    euler = R.from_matrix(transform[:3, :3]).as_euler('xyz')
    # print(f"origin euler: roll: {euler[0]}, pitch: {euler[1]}, yaw:{euler[2]}")
    new_euler = np.empty(3)
    new_euler[0] = euler[1] + (np.pi/2)
    new_euler[1] = euler[0]
    new_euler[2] = euler[2] - (np.pi/2)
    opengl_rotation =  R.from_euler('xyz', [new_euler[0] , new_euler[1], new_euler[2]]).as_matrix()
    opengltransform[:3, -1] = transform[:3, -1]
    opengltransform[:3, :3] = opengl_rotation

    camera = GetCamerasFromTransforms(opengltransform, camera_info)
    return camera

def GetCamerasFromTransforms(transforms, camera_info):
    
    resolution = camera_info["image_resolution"]
    c = camera_info["c"]
    focal = camera_info["focal"]

    fov = focal_len_to_fov(focal, resolution)
    FovX = fov[0]
    FovY = fov[1]

    # NeRF 'transform_matrix' is a camera-to-world transform
    c2w = np.array(transforms)
    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    c2w[:3, 1:3] *= -1
    # get the world-to-camera transform and set R, T
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]
    height = resolution[0]
    width = resolution[1]

    #bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

    return FakeCam(R=R, T=T, FoVx=FovX, FoVy=FovY, width=width, height=height)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test()
