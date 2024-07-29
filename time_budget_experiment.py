import rospy
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

import yaml
import argparse
from planner import get_planner
from planner.utils import uniform_sampling, xyz_to_view
import numpy as np
import scipy.spatial as spatial
from datetime import datetime
import imageio
import glob
#from dotmap import DotMap
import torch
#from neural_rendering.utils import util
#from neural_rendering.evaluation.pretrained_model import PretrainedModel
import pandas
import torch.nn.functional as F
# Gaussian training
from random import randint
from scene import Scene, GaussianModel
from tqdm import tqdm
from utils.loss_utils import l1_loss, ssim
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, network_gui
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
# training manager
import wandb
from utils.cluster_manager import ClusterStateManager
# NBVS
from active.schema import schema_dict
from lpipsPyTorch import lpips, lpips_func
from utils.image_utils import psnr
import time
import csv

#[cyw]:setup_csv
def setup_csv(path):
    training_time_csv = f"{path}/training_time.csv"
    training_file_exists = os.path.isfile(training_time_csv)
    if not training_file_exists:
        with open(training_time_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['iterations', 'loss', 'times'])
    
    algo_time_csv = f"{path}/algo_time.csv"
    algo_file_exists = os.path.isfile(algo_time_csv)
    if not algo_file_exists:
        with open(algo_time_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['select_idxs', 'times'])

    flying_time_csv = f"{path}/flying_time.csv"
    flying_file_exists = os.path.isfile(flying_time_csv)
    if not flying_file_exists:
        with open(flying_time_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['pose_idxs', 'times'])

    captured_time_csv = f"{path}/captured_time.csv"
    captured_file_exists = os.path.isfile(captured_time_csv)
    if not captured_file_exists:
        with open(captured_time_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['pose_idxs', 'times'])

    return training_time_csv, algo_time_csv, flying_time_csv, captured_time_csv
    
csm = ClusterStateManager()

planner_title = {
    "max_distance": "Max. View Distance",
    "random": "Random",
#    "neural_nbv": "Ours",
}

def save_checkpoint(gaussians, iteration, scene, base_iter=0, save_path=None, save_last=True):
    ckpt_dict = {"model_params": gaussians.capture(), "first_iter": iteration, "train_idx": scene.train_idxs, "base_iter": base_iter}

    if save_last:
        last_path = scene.model_path + "/last.pth"
        print("\n[ITER {}] Saving Checkpoint to {}".format(iteration, last_path))
        torch.save(ckpt_dict, last_path)   

    if save_path is None:
        save_path = scene.model_path + "/chkpnt" + str(iteration) + ".pth"
    print("\n[ITER {}] Saving Checkpoint to {}".format(iteration, save_path))
    torch.save(ckpt_dict, save_path)   

def load_checkpoint(ckpt_path: str, gaussians, scene, opt, ignore_train_idxs=False):
    ckpt_dict = torch.load(ckpt_path)
    (model_params, first_iter, train_idxs) = ckpt_dict["model_params"], ckpt_dict["first_iter"], ckpt_dict["train_idx"]
    gaussians.restore(model_params, opt)
    if not ignore_train_idxs:
        scene.train_idxs = train_idxs

    base_iter = ckpt_dict.get("base_iter", 0)
    return first_iter, base_iter


def setup_random_seed(seed):
    np.random.seed(seed)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    # planning experiment in simulator using baseline planners and our planner

    setup_random_seed(10)
    
    experiment_path = dataset.source_path
    time_budget = args.time_budget
    num_views_needed = args.maximum_view
    
    os.makedirs(experiment_path, exist_ok=True)

    print("---------- planning ----------")
    
    phi = 15 * (np.pi/180)
    theta = 0.0
    radius = 3
    init_view = [phi, theta, radius]

    # [cyw]: setup planner type
    planner_type = args.planner_type

    # find planner configuration file
    print(
        f"---------- {planner_type} planner ----------\n"
    )
    planner_cfg_path = os.path.join(
        "planner/config", f"{planner_type}_planner.yaml"
    )
    assert os.path.exists(planner_cfg_path)
    with open(planner_cfg_path, "r") as config_file:
        planner_cfg = yaml.safe_load(config_file)

    planner_cfg.update(args.__dict__)
    planner_cfg["planner_type"] = planner_type
    planner_cfg["experiment_path"] = experiment_path
    planner_cfg["experiment_id"] = args.experiment_id #[cyw]: 
    print(planner_cfg)
    nbv_planner = get_planner(planner_cfg)
    num_views_needed -= 1 
    #[cyw]:flying time
    flying_start_time = time.time()
    nbv_planner.start(initial_view=init_view)
    flying_end_time = time.time()

    #[cyw]:captured time
    captured_start_time = time.time()
    nbv_planner.store_train_set()
    captured_end_time = time.time()

    time_budget = time_budget - (captured_end_time - flying_start_time)
    
    if args.planner_type != "ours" :
        nbv_planner.del_init_view(init_view)
    nbv_planner.store_test_set()
    # nbv = nbv_planner.plan_next_view()
    # nbv_planner.move_sensor(nbv)


    dataset.source_path = nbv_planner.get_record_path()
    print(f"dataset.source_path: {dataset.source_path}" )
    #[cyw]: Gaussian training
    first_iter = 0
    base_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    #[cyw]: setup schema
    # schema = schema_dict[args.schema](dataset_size=len(scene.getTrainCameras()), scene=scene,
    #                                   N=args.maximum_view, M=args.add_view, iteration_base=args.iteration_base,
    #                                   num_init_views=args.num_init_views, interval_epochs=args.interval_epochs,
    #                                   save_ply_each_time=args.save_ply_each_time, save_ply_after_last_adding=args.save_ply_after_last_adding)
    # print(f"schema: {schema.load_its}")
    #[cyw]: change to saving ply when adding the view
    # saving_iterations = list(set(saving_iterations+schema.schema_ckpt))
    # print(saving_iterations)

    # print(f"scene.train_idxs: {scene.train_idxs}")
    init_ckpt_path = f"{args.model_path}/init.ckpt"
    if checkpoint: # this is to continue training in SLURM after requeue
        if os.path.exists(checkpoint):
            first_iter, base_iter = load_checkpoint(checkpoint, gaussians, scene, opt)
        else:
            print(f"[WARNING] checkpoint {checkpoint} doesn't exist, training from scratch")

    if first_iter == 0: # maybe init_ckpt has been save if preempted
        save_checkpoint(gaussians, first_iter, scene, base_iter, save_path=init_ckpt_path, save_last=False)

    #[cyw]:write csv
    training_time_csv, algo_time_csv, flying_time_csv, captured_time_csv = setup_csv(args.model_path)
    pose_idxs = 0 
    with open(flying_time_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([pose_idxs, flying_end_time - flying_start_time])

    with open(captured_time_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([pose_idxs, captured_end_time - captured_start_time])    
    

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    trainig_start_time = time.time()
    for iteration in range(first_iter, opt.iterations + 1):        
        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        
        trainig_end_time = time.time()
        # [cyw]: time_budget 
        if (time_budget > 0) and ((trainig_end_time - trainig_start_time) > 10) and num_views_needed > 0:
            try:
                if args.save_ply_each_time:
                    scene.save(iteration)
                print(f"Start choose the nbv after ITER { iteration}")
                # For sectioned training
                # candidate_views_filter = getattr(schema, "candidate_views_filter")[iteration] if hasattr(schema, "candidate_views_filter") else None
                # scene.candidate_views_filter = candidate_views_filter
                # Because selection is time consumeing ({cyw}:uncertainty estimation callback function)
                num_views = 1
                # [cyw]: select nbv          
                # [cyw]:algorithm time
                algo_start = time.time()
                if args.planner_type == "ours" or args.planner_type == "fisher":
                    nbv = nbv_planner.plan_next_view(gaussians, scene, num_views, pipe, background, exit_func=csm.should_exit)
                else:
                    nbv = nbv_planner.plan_next_view()
                algo_end = time.time()

                flying_start_time = time.time()
                nbv_planner.move_sensor(nbv)
                flying_end_time = time.time()
                print(f"ITER {iteration}: NBV: {nbv}")

                #[cyw]:captured time
                captured_start_time = time.time()
                nbv_planner.store_train_set()
                captured_end_time = time.time()
                pose_idxs += 1
                with open(training_time_csv, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([iteration, ema_loss_for_log, trainig_end_time - trainig_start_time])

                with open(algo_time_csv, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([pose_idxs, algo_end - algo_start])

                with open(flying_time_csv, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([pose_idxs, flying_end_time - flying_start_time])

                with open(captured_time_csv, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([pose_idxs, captured_end_time - captured_start_time])    

                time_budget = time_budget - (captured_end_time - trainig_start_time)
                num_views_needed -= 1 
                trainig_start_time = time.time()
                
            except RuntimeError as e:
                print(e)
                print("selector exited early")
                # NOTE: we use iteration - 1 because the selector is not done
                save_checkpoint(gaussians, iteration - 1, scene)
                csm.requeue()

            #print(f"ITER {iteration}: selected views: {selected_views}")

            scene.train_idxs.extend([len(scene.train_idxs)])
            scene.add_new_cameras(args, nbv_planner.record_path)
            
            print(f"ITER {iteration}: training views after selection: {scene.train_idxs}")

            gaussians.optimizer.zero_grad(set_to_none = True)

            first_iter, _ = load_checkpoint(init_ckpt_path, gaussians, scene, opt, ignore_train_idxs=True)
            base_iter = iteration - 1

        iter_start.record()

        gaussians.update_learning_rate(iteration - base_iter)

        if iteration > args.sh_up_after and iteration % args.sh_up_every == 0:
            gaussians.oneupSHdegree()
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        # print(f"viewpoint_stack: {viewpoint_stack}")
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        # We save before logging
        if csm.should_exit():
            save_checkpoint(gaussians, iteration - 1, scene)
            csm.requeue()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # before_selection = schema.num_views_to_add(iteration + 1) > 0
            before_selection = False
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
                            testing_iterations, scene, render, (pipe, background), before_selection=before_selection, 
                            log_every_image=args.log_every_image)
            if (iteration in saving_iterations): 
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            cur_iter = iteration - base_iter
            if cur_iter < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if cur_iter > opt.densify_from_iter and cur_iter % opt.densification_interval == 0:
                    size_threshold = 20 if cur_iter > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, args.min_opacity, scene.cameras_extent, size_threshold)
                
                if cur_iter % opt.opacity_reset_interval == 0 or (dataset.white_background and cur_iter == opt.densify_from_iter):
                    print(f"\nreset_opacity at {cur_iter}, base_iter")
                    gaussians.reset_opacity() #[cyw]

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
        
        if (iteration in checkpoint_iterations):
            save_checkpoint(gaussians, iteration, scene)
    #[cyw]:training time
    trainig_end_time = time.time()
    with open(training_time_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([iteration, ema_loss_for_log, trainig_end_time - trainig_start_time])
    wandb.finish()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, before_selection=False, log_every_image=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations or before_selection:
        print(f"Running evaluation for iteration: {iteration}")
        torch.cuda.empty_cache()
        lpips = lpips_func("cuda", net_type='vgg')
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0

                log_images = {}
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and ((idx < 5) or log_every_image):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(idx), image[None], global_step=iteration)
                        log_images[f"render/{idx:03d}"] = wandb.Image(image[None])
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(idx), gt_image[None], global_step=iteration)
                            log_images[f"gt/{idx:03d}"] = wandb.Image(gt_image.cpu()[None])
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips.to(image.device)
                    lpips_test += lpips(image, gt_image).mean().double()

                if log_every_image:
                    wandb.log(log_images, step=iteration)

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                log_dict = {config['name'] + '/l1_loss': l1_test, config['name'] + '/psnr': psnr_test,
                            config['name'] + '/ssim': ssim_test, config['name'] + '/lpips': lpips_test,}
                wandb.log(log_dict, step=iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
            wandb.log({'total_points': scene.gaussians.get_xyz.shape[0]}, step=iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":

    rospy.init_node("simulator_experiment")

    parser = ArgumentParser(description="Training script parameters")
    
    #[cyw]: planner_type options: all, random, fisher, ours
    parser.add_argument(
        "--planner_type", "-P", type=str, default="ours", help="planner_type"
    )

    parser.add_argument(
        "--model_name",
        "-M",
        type=str,
        #required=True,
        help="model name of pretrained model",
    )

    # parser.add_argument(
    #     "--test_data_path",
    #     "-TD",
    #     type=str,
    #     required=True,
    #     help="data path",
    # )

    # mandatory arguments
    parser.add_argument(
        "--experiment_id",
        type=int,
        default=1,
        help="experiment id",
    )
    # arguments with default values
    parser.add_argument(
        "--nviews", "-nv", type=int, default=5, help="number of reference views"
    )
    parser.add_argument(
        "--planning_budget",
        "-BG",
        type=int,
        default=20,
        help="maximal measurments for the mission",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="config file path",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="gpu to use, space delimited",
    )

    parser.add_argument(
        "--evaluation_only", action="store_true", help="evaluation mode"
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        default="not defined",
        help="must be defined in evaluation mode",
    )
    # 3DGS args
    lp = ModelParams(parser) #args: model_path
    op = OptimizationParams(parser) #args: iterations
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[15_000, 20_000, 25_000, 30_000]) # the iteration that evaluate metrics
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000]) # the iteration that save 3DGS.ply
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[]) # the iteration that save 3DGS ckpt
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    # fisherrf
    parser.add_argument("--schema", type=str, default="all")
    parser.add_argument("--reg_lambda", type=float, default=1e-6)
    parser.add_argument("--sh_up_every", type=int, default=1_000, help="increase spherical harmonics every N iterations")  #[cyw]:fisher:5000, original 3DGS:1000
    parser.add_argument("--sh_up_after", type=int, default=-1, help="start to increate active_sh_degree after N iterations")   #[cyw]:fisher:-1, original 3DGS:-1
    parser.add_argument("--min_opacity", type=float, default=0.005, help="min_opacity to prune") #[cyw]:fisher:0.005, original 3DGS:0.005
    parser.add_argument("--log_every_image", action="store_true", help="log every images during traing")
    parser.add_argument("--filter_out_grad", nargs="+", type=str, default=["rotation"])

    # experiment
    parser.add_argument("--iteration_base", type=int, default=2000)
    parser.add_argument("--num_init_views", type=int, default=1)
    parser.add_argument("--interval_epochs", type=int, default=100)
    parser.add_argument("--maximum_view", type=int, default=12)
    parser.add_argument("--add_view", type=int, default=1)
    parser.add_argument("--save_ply_each_time", action="store_true")
    parser.add_argument("--save_ply_after_last_adding", action="store_true")
    parser.add_argument("--time_budget", type=float, default=10000.0, help="time budget")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args = parser.parse_args()

    print("Optimizing " + args.model_path)

    wandb.init(project='Ours', resume="allow", id=os.path.split(args.model_path.rstrip('/'))[-1], config=vars(args))
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
