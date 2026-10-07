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
import torch
from random import randint

import torch.nn.functional as F

from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene
from models import (
    optimizationParamTypeCallbacks,
    gaussianModel
)

from utils.general_utils import safe_state
from utils.loss_utils import penalize_outside_range
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams
import matplotlib.pyplot as plt
import json
import time
import numpy as np
import copy



try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def norm_gauss(m, sigma, t):
    log = ((m - t)**2 / sigma**2) / -2
    return torch.exp(log)

def interpolate(A, B, alpha):
    I = (1 - alpha) * A + alpha * B
    return I

def training(gs_type, dataset: ModelParams, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, save_xyz, use_dff):
    time_start = time.process_time()
    init_time = time.time()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    frames = len(os.listdir(f'{dataset.source_path}/original'))
    print("frames", frames)
    gaussians = gaussianModel[gs_type](dataset.sh_degree, dataset.poly_degree, frames, use_dff = use_dff)

    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    bg = torch.rand((3), device="cuda") if opt.random_background else background

    viewpoint_stack = None

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0 
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    viewpoint_cameras = scene.getTrainCameras()
    num_frames = len(viewpoint_cameras)

    gts = []
    for camera in viewpoint_cameras:
        gt = camera.get_image(bg, opt.random_background).cuda()
        gts.append(gt)

    prev_next_overlap = 2 if dataset.camera == "mirror" else 1
    print("prev_next_overlap", prev_next_overlap)

    interpolation = 1

    for iteration in range(first_iter, opt.iterations + 1):
        
        os.makedirs(f"{scene.model_path}/xyz", exist_ok=True)
        if save_xyz and (iteration % 5000 == 1 or iteration == opt.iterations):
            torch.save(gaussians.get_xyz, f"{scene.model_path}/xyz/{iteration}.pt")

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        idx = randint(0, len(viewpoint_cameras) - 1)

        camera_prev = viewpoint_cameras[idx - prev_next_overlap] if idx - prev_next_overlap >= 0 else None
        camera = viewpoint_cameras[idx]
        camera_next = viewpoint_cameras[idx + prev_next_overlap] if idx + prev_next_overlap <= num_frames - 1 else None

        cameras = {"camera_prev": camera_prev ,"camera": camera, "camera_next": camera_next}

        if (iteration - 1) == debug_from:
            pipe.debug = True

        outputs = {}
        outputs_inter = {}
        if iteration == 10000:
            interpolation = 2
            print("changed interpolation")

        for name, camera in cameras.items():
            alpha = np.random.uniform(0.2, 0.8)
            if not camera:
                continue

            if name == "camera_prev":
                A = gts[idx-prev_next_overlap]
                B = gts[idx]
                interpolated_img = interpolate(A, B, alpha)
            elif name == "camera":
                A = gts[idx]
                if idx + prev_next_overlap < len(gts):
                    B = gts[idx + prev_next_overlap]
                    interpolated_img = interpolate(A, B, alpha)
                else:
                    render_pkg = render(camera, gaussians, pipe, bg, train=True, iter=iteration)
                    render_pkg["gt"] = gts[idx]
                    outputs[name] = render_pkg
                    continue
            elif name == "camera_next":
                render_pkg = render(camera, gaussians, pipe, bg, train=True, iter=iteration)
                render_pkg["gt"] = gts[idx + prev_next_overlap]
                outputs[name] = render_pkg
                continue

            for i in range(interpolation):
                if i == 0:
                    render_pkg = render(camera, gaussians, pipe, bg, train=True, iter=iteration)
                    render_pkg["gt"] = A
                    outputs[name] = render_pkg
                else:
                    render_pkg = render(camera, gaussians, pipe, bg, train=True, iter=iteration, alpha=alpha)
                    render_pkg["gt"] = interpolated_img
                    outputs_inter[f"{name}_interpolate_{i}"] = render_pkg


        data = {}
        data_inter = {}

        for k in outputs["camera"].keys():
            if k == "viewspace_points":
                data[k] = [output[k] for output in outputs.values()]
                data_inter[k] = [output[k] for output in outputs_inter.values()]
            elif k in ["visibility_filter", "radii"]:
                data[k] = [output[k] for output in outputs.values()]
                data_inter[k] = [output[k] for output in outputs_inter.values()]
            elif k in ["render", "gt", "mask"]:
                data[k] = torch.stack([output[k] for output in outputs.values()], dim=0)
                if interpolation > 1:
                    data_inter[k] = torch.stack([output[k] for output in outputs_inter.values()], dim=0)


        sigma = gaussians.get_sigma
        if iteration < 20000:
            sigma_loss = penalize_outside_range(sigma, 2./num_frames, 1)
        else:
            sigma_loss = 0.0

        render_curr = outputs["camera"]["render"]
        gt_curr = outputs["camera"]["gt"]
        Ll1 = l1_loss(data["render"], data["gt"])
        if interpolation > 1:
            Ll1_inter = l1_loss(data_inter["render"], data_inter["gt"])
        else:
            Ll1_inter = 0

        data['mask_t'] = torch.stack(data['visibility_filter'], dim=-1).any(dim=1)
        ssim_loss = 1.0 - ssim(data['render'], data['gt'])
        L_flow_temporal = torch.tensor(0.0, device='cuda', dtype=data['render'].dtype)
        # #
        gt_prev_warped = None
        gt_next_warped = None

        loss = 2.0 * Ll1 + 0.5 * Ll1_inter +  0.25 * sigma_loss + 0.25 * ssim_loss

        psnr_ = psnr(render_curr, gt_curr).mean().double()
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background), L_flow_temporal, render_curr, gt_curr, gt_prev_warped, gt_next_warped, gaussians.get_sigma)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            if iteration < 20000:
                radii_batch = torch.stack(data['radii'], dim=-1).max(dim=-1)[0]
                visibility_filter_batch = data["mask_t"]
                gaussians.max_radii2D[visibility_filter_batch] = torch.max(
                    gaussians.max_radii2D[visibility_filter_batch],
                    radii_batch[visibility_filter_batch]
                )
                xyscreen = data['viewspace_points']
                gaussians.add_densification_stats(xyscreen, visibility_filter_batch)
                    
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    print("\n[ITER {}] Densifying Gaussians".format(iteration))
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene.cameras_extent,
                                                size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    print("\n[ITER {}] Reset opacity ".format(iteration))
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    time_elapsed = time.process_time() - time_start
    time_dict = {}
    time_dict["time"] = time_elapsed
    time_dict["elapsed"] = time.time() - init_time
    with open(scene.model_path + f"/time.json", 'w') as fp:
        json.dump(time_dict, fp, indent=True)

def training_binary_segmentation(
    gs_type,
    dataset: ModelParams,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    save_xyz,
    use_dff,
    seg_head_only: bool = False,   # NEW
    ):
    time_start = time.process_time()
    init_time = time.time()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    frames = len(os.listdir(f'{dataset.source_path}/original'))
    print("frames", frames)

    gaussians = gaussianModel[gs_type](dataset.sh_degree, dataset.poly_degree, frames, use_dff=use_dff)
    scene = Scene(dataset, gaussians, shuffle=False)

    # Standard training setup (will be overridden for seg_head_only later)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # Segmentation-head-only Stage-2 mode:
    # - geometry + img head are frozen
    # - seg head params initialized from img head
    # - optimizer rebuilt on seg head only
    if seg_head_only:
        first_iter = 0
        print("Segmentation head only mode: freezing geometry & img head, optimizing segmentation head.")
        gaussians.init_seg_head_from_img()
        gaussians.freeze_geometry_and_img_head()
        seg_params = gaussians.seg_head_parameters()
        gaussians.optimizer = torch.optim.Adam(
            [
                {
                    "params": seg_params,
                    "lr": opt.feature_lr,
                    "name": "seg_head",
                }
            ],
            lr=0.0,
            eps=1e-15,
        )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    bg = torch.rand((3), device="cuda") if opt.random_background else background

    viewpoint_stack = None

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    viewpoint_cameras = scene.getTrainCameras()
    num_frames = len(viewpoint_cameras)

    gts = []
    for camera in viewpoint_cameras:
        gt = camera.get_image(bg, opt.random_background).cuda()
        gts.append(gt)

    prev_next_overlap = 2 if dataset.camera == "mirror" else 1
    print("prev_next_overlap", prev_next_overlap)

    interpolation = 1
    head_name = "seg" if seg_head_only else "img"   # NEW: which appearance head to use

    for iteration in range(first_iter, opt.iterations + 1):
        os.makedirs(f"{scene.model_path}/xyz", exist_ok=True)
        if save_xyz and (iteration % 5000 == 1 or iteration == opt.iterations):
            torch.save(gaussians.get_xyz, f"{scene.model_path}/xyz/{iteration}.pt")

        iter_start.record()

        # For seg_head_only we keep constant LR (no xyz group in optimizer),
        # but calling update_learning_rate is harmless because it only
        # touches the group named "xyz", which we didn't define in that mode.
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        idx = randint(0, len(viewpoint_cameras) - 1)

        camera_prev = viewpoint_cameras[idx - prev_next_overlap] if idx - prev_next_overlap >= 0 else None
        camera = viewpoint_cameras[idx]
        camera_next = viewpoint_cameras[idx + prev_next_overlap] if idx + prev_next_overlap <= num_frames - 1 else None

        cameras = {"camera_prev": camera_prev, "camera": camera, "camera_next": camera_next}

        if (iteration - 1) == debug_from:
            pipe.debug = True

        outputs = []
        outputs_inter = {}

        for name, cam in cameras.items():
            if not cam:
                continue
            render_pkg = render(
                cam,
                gaussians,
                pipe,
                bg,
                train=True,
                iter=iteration,
                seg=True,
                head=head_name,   
            )
            render_pkg["gt"] = cam.get_image(bg, opt.random_background).cuda()
            outputs.append(render_pkg)

        data = {}
        for k in outputs[0].keys():
            if k == "viewspace_points":
                data[k] = [output[k] for output in outputs]
            elif k in ["visibility_filter", "radii"]:
                data[k] = [output[k] for output in outputs]
            elif k in ["render", "gt", "mask"]:
                data[k] = torch.stack([output[k] for output in outputs], dim=0)

        sigma = gaussians.get_sigma
        if iteration < 20000 and not seg_head_only:
            # In seg_head_only, geometry (including sigma) is frozen,
            # so we do not regularize it further.
            sigma_loss = penalize_outside_range(sigma, 2.0 / num_frames, 1)
        else:
            sigma_loss = 0.0

        render_curr = outputs[0]["render"]
        gt_curr = outputs[0]["gt"]
        Ll1 = l1_loss(data["render"], data["gt"])

        data["mask_t"] = torch.stack(data["visibility_filter"], dim=-1).any(dim=1)
        ssim_loss = 1.0 - ssim(data["render"], data["gt"])
        L_flow_temporal = torch.tensor(0.0, device="cuda", dtype=data["render"].dtype)
        gt_prev_warped = None
        gt_next_warped = None


        loss = 2.0 * Ll1 + 0.5 * sigma_loss

        psnr_ = psnr(data["render"], data["gt"]).mean().double()
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 100 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "psnr": f"{psnr_:.{2}f}",
                        "point": f"{total_point}",
                    }
                )
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()


            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                L_flow_temporal,
                render_curr,
                gt_curr,
                gt_prev_warped,
                gt_next_warped,
                gaussians.get_sigma,
                seg=True,
                head=head_name,  
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification and opacity reset: disable in seg_head_only mode
            if iteration < 20000 and not seg_head_only:
                radii_batch = torch.stack(data["radii"], dim=-1).max(dim=-1)[0]
                visibility_filter_batch = data["mask_t"]
                gaussians.max_radii2D[visibility_filter_batch] = torch.max(
                    gaussians.max_radii2D[visibility_filter_batch],
                    radii_batch[visibility_filter_batch],
                )
                xyscreen = data["viewspace_points"]
                gaussians.add_densification_stats(xyscreen, visibility_filter_batch)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    print("\n[ITER {}] Densifying Gaussians".format(iteration))
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.01,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    print("\n[ITER {}] Reset opacity ".format(iteration))
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    time_elapsed = time.process_time() - time_start
    time_dict = {}
    time_dict["time"] = time_elapsed
    time_dict["elapsed"] = time.time() - init_time
    with open(scene.model_path + f"/time.json", "w") as fp:
        json.dump(time_dict, fp, indent=True)

def training_joint(
    gs_type,
    dataset_img: ModelParams,
    dataset_seg: ModelParams,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    save_xyz,
    use_dff,
    lambda_img: float = 1.0,
    lambda_seg: float = 1.0,
):
    """
    One-stage joint training:
      - shared Gaussians (geometry + temporal params)
      - two heads:
          img head -> photometric loss (seg=False, head="img")
          seg head -> segmentation loss (seg=True,  head="seg")
    """

    time_start = time.process_time()
    init_time = time.time()
    first_iter = 0

    # Use img dataset for output folder
    tb_writer = prepare_output_and_logger(dataset_img)

    # Make seg dataset write to same folder (so plots/checkpoints go to one place)
    dataset_seg.model_path = dataset_img.model_path

    frames_img = len(os.listdir(f"{dataset_img.source_path}/original"))
    frames_seg = len(os.listdir(f"{dataset_seg.source_path}/original"))
    print("frames(img)", frames_img)
    print("frames(seg)", frames_seg)
    assert frames_img == frames_seg, "USG and mask datasets must have same number of frames"
    frames = frames_img

    # Shared gaussians trained jointly
    gaussians = gaussianModel[gs_type](dataset_img.sh_degree, dataset_img.poly_degree, frames, use_dff=use_dff)

    # Scene for IMG initializes gaussians from PCD/PLY and provides cameras
    scene_img = Scene(dataset_img, gaussians, shuffle=False)

    # Build SEG cameras via a dummy gaussian model so we don't reinitialize the shared gaussians
    gaussians_dummy = gaussianModel[gs_type](dataset_seg.sh_degree, dataset_seg.poly_degree, frames, use_dff=use_dff)
    scene_seg = Scene(dataset_seg, gaussians_dummy, shuffle=False)
    scene_seg.gaussians = gaussians


    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)

        # Detect whether checkpoint already contains a seg head + optimizer state (joint resume)
        ckpt_has_seg = False
        ckpt_has_opt = False
        if isinstance(model_params, dict):
            op_seg = model_params.get("_opacity_seg", None)
            ckpt_has_seg = (op_seg is not None) and hasattr(op_seg, "numel") and (op_seg.numel() > 0)
            ckpt_has_opt = model_params.get("optimizer_state", None) is not None

        if ckpt_has_seg and ckpt_has_opt:
            # Proper joint checkpoint: restore + load optimizer state
            print(f"[joint] Resuming full joint checkpoint (with seg head + optimizer) from: {checkpoint}")
            gaussians.restore(model_params, training_args=opt, load_optimizer=True)
        else:
            # Stage-1 checkpoint or older format: load weights only, then create seg head and fresh optimizer
            print(f"[joint] Loading weights only (no optimizer) from: {checkpoint}")
            gaussians.restore(model_params, training_args=None, load_optimizer=False)

            # Create seg head now (must exist BEFORE training_setup so optimizer has seg groups)
            gaussians.init_seg_head_from_img()

            # Fresh optimizer (will include seg groups as separate param groups if gaussian_model.py was updated)
            gaussians.training_setup(opt)

            # Start joint training from iteration 0 (optimizer is new)
            first_iter = 0
    else:
        # No checkpoint: initialize seg head from initial img head, then setup optimizer
        gaussians.init_seg_head_from_img()
        gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset_img.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    ema_psnr_img_for_log = 0.0
    ema_psnr_seg_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    cams_img = scene_img.getTrainCameras()
    cams_seg = scene_seg.getTrainCameras()
    assert len(cams_img) == len(cams_seg), "Camera lists must match between datasets"
    num_frames = len(cams_img)


    gts_img = [cam.get_image(bg, opt.random_background).cuda() for cam in cams_img]
    gts_seg = [cam.get_image(bg, opt.random_background).cuda() for cam in cams_seg]

    prev_next_overlap = 2 if dataset_img.camera == "mirror" else 1
    print("prev_next_overlap", prev_next_overlap)

    interpolation = 1

    for iteration in range(first_iter, opt.iterations + 1):
        os.makedirs(f"{scene_img.model_path}/xyz", exist_ok=True)
        if save_xyz and (iteration % 5000 == 1 or iteration == opt.iterations):
            torch.save(gaussians.get_xyz, f"{scene_img.model_path}/xyz/{iteration}.pt")

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if (iteration - 1) == debug_from:
            pipe.debug = True

        if iteration == 10000:
            interpolation = 2
            print("changed interpolation")

        idx = randint(0, num_frames - 1)


        cam_prev_img = cams_img[idx - prev_next_overlap] if idx - prev_next_overlap >= 0 else None
        cam_img = cams_img[idx]
        cam_next_img = cams_img[idx + prev_next_overlap] if idx + prev_next_overlap <= num_frames - 1 else None
        cameras_img = {"camera_prev": cam_prev_img, "camera": cam_img, "camera_next": cam_next_img}

        outputs_img = {}
        outputs_img_inter = {}

        for name, cam in cameras_img.items():
            alpha = np.random.uniform(0.2, 0.8)
            if cam is None:
                continue

            if name == "camera_prev":
                A = gts_img[idx - prev_next_overlap]
                B = gts_img[idx]
                interpolated_img = interpolate(A, B, alpha)

            elif name == "camera":
                A = gts_img[idx]
                if idx + prev_next_overlap < len(gts_img):
                    B = gts_img[idx + prev_next_overlap]
                    interpolated_img = interpolate(A, B, alpha)
                else:
                    rpkg = render(cam, gaussians, pipe, bg, train=True, iter=iteration, seg=False, head="img")
                    rpkg["gt"] = gts_img[idx]
                    outputs_img[name] = rpkg
                    continue

            elif name == "camera_next":
                rpkg = render(cam, gaussians, pipe, bg, train=True, iter=iteration, seg=False, head="img")
                rpkg["gt"] = gts_img[idx + prev_next_overlap]
                outputs_img[name] = rpkg
                continue

            for i in range(interpolation):
                if i == 0:
                    rpkg = render(cam, gaussians, pipe, bg, train=True, iter=iteration, seg=False, head="img")
                    rpkg["gt"] = A
                    outputs_img[name] = rpkg
                else:
                    rpkg = render(cam, gaussians, pipe, bg, train=True, iter=iteration, seg=False, head="img", alpha=alpha)
                    rpkg["gt"] = interpolated_img
                    outputs_img_inter[f"{name}_interpolate_{i}"] = rpkg

        data_img = {}
        data_img_inter = {}
        for k in outputs_img["camera"].keys():
            if k == "viewspace_points":
                data_img[k] = [o[k] for o in outputs_img.values()]
                data_img_inter[k] = [o[k] for o in outputs_img_inter.values()]
            elif k in ["visibility_filter", "radii"]:
                data_img[k] = [o[k] for o in outputs_img.values()]
                data_img_inter[k] = [o[k] for o in outputs_img_inter.values()]
            elif k in ["render", "gt", "mask"]:
                data_img[k] = torch.stack([o[k] for o in outputs_img.values()], dim=0)
                if interpolation > 1 and len(outputs_img_inter) > 0:
                    data_img_inter[k] = torch.stack([o[k] for o in outputs_img_inter.values()], dim=0)

        sigma = gaussians.get_sigma
        if iteration < 20000:
            sigma_loss = penalize_outside_range(sigma, 2.0 / num_frames, 1)
        else:
            sigma_loss = 0.0

        render_img_curr = outputs_img["camera"]["render"]
        gt_img_curr = outputs_img["camera"]["gt"]

        Ll1_img = l1_loss(data_img["render"], data_img["gt"])
        Ll1_img_inter = l1_loss(data_img_inter["render"], data_img_inter["gt"]) if (interpolation > 1 and "render" in data_img_inter) else 0.0
        ssim_loss = 1.0 - ssim(data_img["render"], data_img["gt"])

        loss_img = 2.0 * Ll1_img + 0.5 * Ll1_img_inter + 0.25 * sigma_loss + 0.25 * ssim_loss
        psnr_img = psnr(render_img_curr, gt_img_curr).mean().double()


        cam_prev_seg = cams_seg[idx - prev_next_overlap] if idx - prev_next_overlap >= 0 else None
        cam_seg = cams_seg[idx]
        cam_next_seg = cams_seg[idx + prev_next_overlap] if idx + prev_next_overlap <= num_frames - 1 else None
        cameras_seg = {"camera_prev": cam_prev_seg, "camera": cam_seg, "camera_next": cam_next_seg}

        outputs_seg = {}
        for name, cam in cameras_seg.items():
            if cam is None:
                continue
            rpkg = render(cam, gaussians, pipe, bg, train=True, iter=iteration, seg=True, head="seg")
            if name == "camera_prev":
                rpkg["gt"] = gts_seg[idx - prev_next_overlap]
            elif name == "camera":
                rpkg["gt"] = gts_seg[idx]
            else:
                rpkg["gt"] = gts_seg[idx + prev_next_overlap]
            outputs_seg[name] = rpkg

        data_seg = {}
        for k in outputs_seg["camera"].keys():
            if k == "viewspace_points":
                data_seg[k] = [o[k] for o in outputs_seg.values()]
            elif k in ["visibility_filter", "radii"]:
                data_seg[k] = [o[k] for o in outputs_seg.values()]
            elif k in ["render", "gt", "mask"]:
                data_seg[k] = torch.stack([o[k] for o in outputs_seg.values()], dim=0)

        Ll1_seg = l1_loss(data_seg["render"], data_seg["gt"])
        loss_seg = 2.0 * Ll1_seg + 0.5 * sigma_loss
        render_seg_curr = outputs_seg["camera"]["render"]
        gt_seg_curr = outputs_seg["camera"]["gt"]
        psnr_seg = psnr(render_seg_curr, gt_seg_curr).mean().double()

        # 
        # Joint loss + backward
        # 
        loss = lambda_img * loss_img + lambda_seg * loss_seg
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_img_for_log = 0.4 * psnr_img + 0.6 * ema_psnr_img_for_log
            ema_psnr_seg_for_log = 0.4 * psnr_seg + 0.6 * ema_psnr_seg_for_log

            total_point = gaussians._xyz.shape[0]
            if iteration % 100 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.7f}",
                        "psnr_img": f"{psnr_img:.2f}",
                        "psnr_seg": f"{psnr_seg:.2f}",
                        "point": f"{total_point}",
                    }
                )
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()

            # Reports / plots (separate prefixes)
            L_flow_temporal = torch.tensor(0.0, device="cuda", dtype=data_img["render"].dtype)

            training_report(
                tb_writer,
                iteration,
                Ll1_img,
                loss_img,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene_img,
                render,
                (pipe, background),
                L_flow_temporal,
                render_img_curr,
                gt_img_curr,
                None,
                None,
                gaussians.get_sigma,
                seg=False,
                head="img",
                plot_prefix="img",
            )

            training_report(
                tb_writer,
                iteration,
                Ll1_seg,
                loss_seg,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene_seg, 
                render,
                (pipe, background),
                L_flow_temporal,
                render_seg_curr,
                gt_seg_curr,
                None,
                None,
                gaussians.get_sigma,
                seg=True,
                head="seg",
                plot_prefix="seg",
            )

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene_img.save(iteration)

            
            if iteration < 20000:
                mask_img = torch.stack(data_img["visibility_filter"], dim=-1).any(dim=1)
                radii_img = torch.stack(data_img["radii"], dim=-1).max(dim=-1)[0]
                xyscreen_img = data_img["viewspace_points"]

                mask_seg = torch.stack(data_seg["visibility_filter"], dim=-1).any(dim=1)
                radii_seg = torch.stack(data_seg["radii"], dim=-1).max(dim=-1)[0]
                xyscreen_seg = data_seg["viewspace_points"]

                gaussians.max_radii2D[mask_img] = torch.max(gaussians.max_radii2D[mask_img], radii_img[mask_img])
                gaussians.max_radii2D[mask_seg] = torch.max(gaussians.max_radii2D[mask_seg], radii_seg[mask_seg])

                gaussians.add_densification_stats(xyscreen_img, mask_img)
                gaussians.add_densification_stats(xyscreen_seg, mask_seg)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene_img.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset_img.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene_img.model_path + f"/chkpnt{iteration}.pth")

    time_elapsed = time.process_time() - time_start
    time_dict = {"time": time_elapsed, "elapsed": time.time() - init_time}
    with open(scene_img.model_path + "/time.json", "w") as fp:
        json.dump(time_dict, fp, indent=True)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l2_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, temporal_loss, render_curr, gt_curr, gt_prev_warped, gt_next_warped, sigma,
                    seg=False, head="img", plot_prefix="img"):

    if tb_writer:
        tb_writer.add_scalar(f'{plot_prefix}/train_l1', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{plot_prefix}/train_loss_total', loss.item(), iteration)
        tb_writer.add_scalar(f'{plot_prefix}/iter_time', elapsed, iteration)
        tb_writer.add_scalar(f'{plot_prefix}/train_temporal_loss', temporal_loss.item(), iteration)

    if tb_writer and iteration % 1000 == 0:
        tb_writer.add_images(f"{plot_prefix}/temporal/render_curr", render_curr, global_step=iteration, dataformats="CHW")
        tb_writer.add_images(f"{plot_prefix}/temporal/gt_curr", gt_curr, global_step=iteration, dataformats="CHW")
        if gt_prev_warped is not None:
            tb_writer.add_images(f"{plot_prefix}/temporal/prev_warped", gt_prev_warped, global_step=iteration, dataformats="CHW")
        if gt_next_warped is not None:
            tb_writer.add_images(f"{plot_prefix}/temporal/next_warped", gt_next_warped, global_step=iteration, dataformats="CHW")
        tb_writer.add_histogram(f'{plot_prefix}/sigma', sigma, global_step=iteration, bins='auto')

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        config = {'name': 'test', 'cameras': scene.getTestCameras()}

        l1_test = 0.0
        psnr_test = 0.0
        psnrs = []
        times = []

        for idx, viewpoint in enumerate(config['cameras']):
            image = torch.clamp(
                renderFunc(viewpoint, scene.gaussians, *renderArgs, seg=seg, head=head)["render"],
                0.0, 1.0
            )
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

            if tb_writer and (idx < 5):
                tb_writer.add_images(f"{plot_prefix}/{config['name']}_view_{viewpoint.image_name}/render",
                                     image[None], global_step=iteration)
                if iteration == testing_iterations[0]:
                    tb_writer.add_images(f"{plot_prefix}/{config['name']}_view_{viewpoint.image_name}/ground_truth",
                                         gt_image[None], global_step=iteration)

            l1_test += l1_loss(image, gt_image).mean().double()
            p = psnr(image, gt_image).mean().double().cpu()
            psnrs.append(p)
            times.append(viewpoint.time)
            psnr_test += p

        psnr_test /= len(config['cameras'])
        l1_test /= len(config['cameras'])

        # ---- save a separate plot per head ----
        plots_dir = os.path.join(scene.model_path, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        plt.plot(times, psnrs, 'o')
        plt.ylabel("PSNR")
        plt.xlabel("Frame")
        plt.savefig(os.path.join(plots_dir, f"{plot_prefix}_{iteration}.png"))
        plt.clf()

        print(f"\n[ITER {iteration}] ({plot_prefix}) Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")

        if tb_writer:
            tb_writer.add_scalar(f'{plot_prefix}/{config["name"]}_l1', l1_test, iteration)
            tb_writer.add_scalar(f'{plot_prefix}/{config["name"]}_psnr', psnr_test, iteration)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--gs_type', type=str, default="gs")
    parser.add_argument('--camera', type=str, default="mirror")
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 40_000, 100_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--save_xyz", action='store_true')
    parser.add_argument("--poly_degree", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--use_dff", type=bool, default=False)
    parser.add_argument("--pipeline", type=str, choices=["img", "seg", "joint"], default="img")
    parser.add_argument("--seg_head_only", action="store_true",
                    help="Freeze geometry & img head, train only segmentation head (requires --start_checkpoint).")
    parser.add_argument("--seg_source_path", type=str, default=None,
                    help="Path to segmentation/mask dataset root (required for --pipeline joint).")
    parser.add_argument("--lambda_img", type=float, default=1.0)
    parser.add_argument("--lambda_seg", type=float, default=1.0)



    lp = ModelParams(parser)
    args, _ = parser.parse_known_args(sys.argv[1:])
    lp.gs_type = args.gs_type
    lp.camera = args.camera
    lp.distance = args.distance
    lp.num_pts = args.num_pts
    lp.poly_degree = args.poly_degree
    
    op = optimizationParamTypeCallbacks[args.gs_type](parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    op.batch_size = args.batch_size

    args.save_iterations.append(args.iterations)

    print("torch cuda: ", torch.cuda.is_available())
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if args.pipeline == "joint":
        if args.seg_source_path is None:
            raise ValueError("--seg_source_path is required for --pipeline joint")

        ds_img = lp.extract(args)
        ds_seg = copy.deepcopy(ds_img)
        ds_seg.source_path = args.seg_source_path

        print("Training with JOINT img+seg loss (no freezing)")
        training_joint(
            args.gs_type,
            ds_img,
            ds_seg,
            op.extract(args),
            pp.extract(args),
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
            args.debug_from,
            args.save_xyz,
            args.use_dff,
            lambda_img=args.lambda_img,
            lambda_seg=args.lambda_seg,
        )

    elif args.pipeline == "seg":
        print("Training with binary segmentation loss")
        training_binary_segmentation(
            args.gs_type,
            lp.extract(args), op.extract(args), pp.extract(args),
            args.test_iterations, args.save_iterations, args.checkpoint_iterations,
            args.start_checkpoint, args.debug_from, args.save_xyz, args.use_dff,
            seg_head_only=args.seg_head_only,
        )
    else:
        print("Training with photometric loss")
        training(
            args.gs_type,
            lp.extract(args), op.extract(args), pp.extract(args),
            args.test_iterations, args.save_iterations, args.checkpoint_iterations,
            args.start_checkpoint, args.debug_from, args.save_xyz, args.use_dff
        )


    print("\nTraining complete.")
    print()
