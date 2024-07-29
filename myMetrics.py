#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# Usage
# python ./myMetrics.py --model_paths ./output/materials_sh0 --gt_model_path ./output/materials_gt

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr, psnr_mask
from argparse import ArgumentParser
import numpy as np


def readImages(renders_dir, gt_dir, mask_dir=None):
    not_exist_list = []
    renders = []
    y_renders = []
    gts = []
    y_gts = []
    masks = []
    image_names = []
    numPixels = []
    
    for fname in os.listdir(gt_dir):  
        gt = Image.open(gt_dir / fname)
        y_gt = gt.convert('YCbCr')
        
        if os.path.exists(renders_dir / fname):
            # print(renders_dir/fname)
            render = Image.open(renders_dir / fname)
        else:
            not_exist_list.append(f"{renders_dir/fname} not exists")
            # print(f"{renders_dir/fname} not exists")
            render = Image.fromarray(np.uint8(np.ones((gt.size[1], gt.size[0], 3))*0))
            
        y_render = render.convert('YCbCr')    
            
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        # print(gt.size)
        gt = scale_image(gt, render.size)
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        y_renders.append(tf.to_tensor(y_render.split()[0]).unsqueeze(0)[:, :3, :, :].cuda())
        y_gts.append(tf.to_tensor(y_gt.split()[0]).unsqueeze(0)[:, :3, :, :].cuda())
        
        if mask_dir:
            mask = Image.open(Path(mask_dir) / fname)
            mask = scale_image(mask, render.size)
            masks.append(tf.to_tensor(mask).unsqueeze(0)[:, :3, :, :].cuda())    
            
        image_names.append(fname)
        numPixels.append(np.count_nonzero(render.split()[-1]))
    
    return renders, gts, masks, image_names, numPixels, y_renders, y_gts, not_exist_list

def scale_image(img, dim):
    # return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img.resize(dim)
    
def evaluate(main_paths, renders_paths, gt_model_paths, with_mask, mask_paths, with_lpips):
    
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    
    gt_full_dict = {}
    gt_per_view_dict = {}
    gt_full_dict_polytopeonly = {}
    gt_per_view_dict_polytopeonly = {}
    
    if not with_mask:
        mask_paths = gt_model_paths # fill in the fake masks path
        
    for main_dir, renders_dir, gt_dir, mask_dir in zip(main_paths, renders_paths, gt_model_paths, mask_paths):
        # main_dir = Path(main_dir)
        renders_dir = Path(renders_dir)
        gt_dir = Path(gt_dir)
        try:
            print("renders_dir:", renders_dir)
            print("gt_dir:", gt_dir)
            if with_mask:
                print("mask_dir:", mask_dir)
                renders, gts, masks, image_names, numPixels, y_renders, y_gts, not_exist_list = readImages(renders_dir, gt_dir, mask_dir)
            else:
                renders, gts, masks, image_names, numPixels, y_renders, y_gts, not_exist_list = readImages(renders_dir, gt_dir)
            print("Finish reading")
            
            ssims = []
            psnrs = []
            psnrs_y = []
            psnr_masks = []
            psnr_y_masks = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                # print(image_names[idx])
                ssims.append(ssim(renders[idx], gts[idx]))
                # print("SSIM")
                psnrs.append(psnr(renders[idx], gts[idx]))
                # print("PSNR")
                psnrs_y.append(psnr(y_renders[idx], y_gts[idx]))
                if with_mask:
                    psnr_masks.append(psnr_mask(renders[idx], gts[idx], masks[idx]))
                    psnr_y_masks.append(psnr_mask(y_renders[idx], y_gts[idx], masks[idx]))
                if with_lpips:
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                
            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  PSNR_Y : {:>12.7f}".format(torch.tensor(psnrs_y).mean(), ".5"))
            if with_mask:
                print("  PSNR_MASK : {:>12.7f}".format(torch.tensor(psnr_masks).mean(), ".5"))
                print("  PSNR_Y_MASK : {:>12.7f}".format(torch.tensor(psnr_y_masks).mean(), ".5"))
            if with_lpips:
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")
            
            results_dict = {
                "SSIM": torch.tensor(ssims).mean().item(),
                "PSNR": torch.tensor(psnrs).mean().item(),
                "PSNR_Y": torch.tensor(psnrs_y).mean().item(),
            }
            per_results_dict = {
                "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                "PSNR_Y": {name: psnr_y for psnr_y, name in zip(torch.tensor(psnrs_y).tolist(), image_names)},                                        
            }
            if with_mask:
                results_dict["PSNR_MASK"] = torch.tensor(psnr_masks).mean().item()
                results_dict["PSNR_Y_MASK"] = torch.tensor(psnr_y_masks).mean().item()
                per_results_dict["PSNR_MASK"] = {name: psnr_mask for psnr_mask, name in zip(torch.tensor(psnr_masks).tolist(), image_names)}
                per_results_dict["PSNR_Y_MASK"] = {name: psnr_y_mask for psnr_y_mask, name in zip(torch.tensor(psnr_y_masks).tolist(), image_names)}
            if with_lpips:
                results_dict["LPIPS"] = torch.tensor(lpipss).mean().item()
                per_results_dict["LPIPS"] = {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}
            
            full_dict.update(results_dict)
            per_view_dict.update(per_results_dict)
            
            # full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
            #                                         "PSNR": torch.tensor(psnrs).mean().item(),
            #                                         "LPIPS": torch.tensor(lpipss).mean().item()})
            # per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
            #                                             "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
            #                                             "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(main_dir + f"/results.json", 'w') as fp:
                json.dump(full_dict, fp, indent=True)
            with open(main_dir + f"/per_view.json", 'w') as fp:
                json.dump(per_view_dict, fp, indent=True)
            with open(main_dir + f"/per_view_notExist.txt", 'w') as f:
                for line in not_exist_list:
                    f.write(f"{line}\n")
        except:
            print("Unable to compute metrics for model", main_dir)

if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--main_dir', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--render_paths', '-r', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--gt_model_paths', '-gt', required=True, nargs="+", type=str, default=[])
    # parser.add_argument('--mode_name', '-mode', required=True, type=str, default="")
    # parser.add_argument('--compare_with', '-compare', required=True, type=str, default="")
    parser.add_argument("--with_mask", action="store_true", default=False) # [YC] add
    parser.add_argument("--mask_paths", '-p', nargs="+", type=str, default=[]) # [YC] add
    parser.add_argument("--with_lpips", action="store_true", default=False) # [YC] add
    parser.add_argument("--torch_device", '-gpu', type=int, default=0) # [YC] add
    args = parser.parse_args()
    
    # device = torch.device(f"cuda:{args.torch_device}")
    # torch.cuda.set_device(device)
    
    evaluate(args.main_dir, args.render_paths, args.gt_model_paths, args.with_mask, args.mask_paths, args.with_lpips)
