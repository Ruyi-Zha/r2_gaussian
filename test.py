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

import os
import os.path as osp
import sys
import torch
from tqdm import tqdm
import torchvision
from time import time
import numpy as np
import concurrent.futures
import yaml
import json
import matplotlib.cm as cm
from argparse import ArgumentParser, Namespace


sys.path.append("./")
from r2_gaussian.arguments import (
    ModelParams,
    PipelineParams,
    OptimizationParams,
    get_combined_args,
)
from r2_gaussian.dataset import Scene
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state
from r2_gaussian.utils.image_utils import metric_vol, metric_proj


def testing(
    dataset: ModelParams,
    pipeline: PipelineParams,
    iteration: int,
    skip_render_train: bool,
    skip_render_test: bool,
    skip_recon: bool,
):
    # Set up dataset

    scene = Scene(
        dataset,
        shuffle=False,
    )

    # Set up some parameters
    volume_to_world = min(scene.scanner_cfg["sVoxel"])
    scale_bound = None
    if dataset.scale_min and dataset.scale_max:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound)
    loaded_iter = initialize_gaussian(gaussians, dataset, iteration)
    scene.gaussians = gaussians

    save_path = osp.join(
        dataset.model_path,
        "test",
        "iter_{}".format(loaded_iter),
    )

    # Evaluate projection train
    if not skip_render_train:
        evaluate_render(
            save_path,
            "render_train",
            scene.getTrainCameras(),
            gaussians,
            pipeline,
        )
    # Evaluate projection test
    if not skip_render_test:
        evaluate_render(
            save_path,
            "render_test",
            scene.getTestCameras(),
            gaussians,
            pipeline,
        )
    # Evaluate volume reconstruction
    if not skip_recon:
        evaluate_volume(
            save_path,
            "reconstruction",
            scene.scanner_cfg,
            gaussians,
            pipeline,
            scene.vol_gt,
        )


def evaluate_volume(
    save_path,
    name,
    scanner_cfg,
    gaussians: GaussianModel,
    pipeline: PipelineParams,
    vol_gt,
):
    """Evaluate volume reconstruction."""
    slice_save_path = osp.join(save_path, name)
    os.makedirs(slice_save_path, exist_ok=True)

    start_time = time()
    query_pkg = query(
        gaussians,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipeline,
    )
    duration = time() - start_time
    vol_pred = query_pkg["vol"]

    psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
    ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")

    multithread_write(
        [vol_gt[..., i][None] for i in range(vol_gt.shape[2])],
        slice_save_path,
        "_gt",
    )
    multithread_write(
        [vol_pred[..., i][None] for i in range(vol_pred.shape[2])],
        slice_save_path,
        "_pred",
    )
    fps = 1 / duration
    eval_dict = {
        "psnr_3d": psnr_3d,
        "ssim_3d": ssim_3d,
        "ssim_3d_x": ssim_3d_axis[0],
        "ssim_3d_y": ssim_3d_axis[1],
        "ssim_3d_z": ssim_3d_axis[2],
        "FPS": fps,
    }

    with open(osp.join(save_path, "eval3d.yml"), "w") as f:
        yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)

    np.save(osp.join(save_path, "vol_gt.npy"), vol_gt.cpu().numpy())
    np.save(osp.join(save_path, "vol_pred.npy"), vol_pred.cpu().numpy())
    print(f"{name} complete. psnr_3d: {psnr_3d}, ssim_3d: {ssim_3d}, fps: {fps}.")


def evaluate_render(save_path, name, views, gaussians, pipeline):
    """Evaluate projection rendering."""
    proj_save_path = osp.join(save_path, name)

    # If already rendered, skip.
    if osp.exists(osp.join(save_path, "eval.yml")):
        print("{} in {} already rendered. Skip.".format(name, save_path))
        return
    os.makedirs(proj_save_path, exist_ok=True)

    gt_list = []
    render_list = []
    time1 = time()
    for view in tqdm(views, desc="render {}".format(name), leave=False):
        rendering = render(view, gaussians, pipeline)["render"]
        gt = view.original_image[0:3, :, :]
        gt_list.append(gt)
        render_list.append(rendering)

    time2 = time()
    fps = (len(views) - 1) / (time2 - time1)
    multithread_write(gt_list, proj_save_path, "_gt")
    multithread_write(render_list, proj_save_path, "_pred")

    images = torch.concat(render_list, 0).permute(1, 2, 0)
    gt_images = torch.concat(gt_list, 0).permute(1, 2, 0)
    psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
    ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
    eval_dict = {
        "psnr_2d": psnr_2d,
        "ssim_2d": ssim_2d,
        "psnr_2d_projs": psnr_2d_projs,
        "ssim_2d_projs": ssim_2d_projs,
        "fps": fps,
    }
    with open(osp.join(save_path, f"eval2d_{name}.yml"), "w") as f:
        yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
    print(
        f"{name} complete. psnr_2d: {eval_dict['psnr_2d']}, ssim_2d: {eval_dict['ssim_2d']}, fps: {fps}."
    )


def multithread_write(image_list, path, suffix):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)

    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(
                image, osp.join(path, "{0:05d}".format(count) + "{}.png".format(suffix))
            )
            np.save(
                osp.join(path, "{0:05d}".format(count) + "{}.npy".format(suffix)),
                image.cpu().numpy()[0],
            )
            return count, True
        except:
            return count, False

    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_render_train", action="store_true", default=False)
    parser.add_argument("--skip_render_test", action="store_true", default=False)
    parser.add_argument("--skip_recon", action="store_true", default=False)
    args = get_combined_args(parser)

    safe_state(args.quiet)

    with torch.no_grad():
        testing(
            model.extract(args),
            pipeline.extract(args),
            args.iteration,
            args.skip_render_train,
            args.skip_render_test,
            args.skip_recon,
        )
