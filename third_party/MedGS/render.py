import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from models import gaussianModelRender, gaussianModel  # NOTE: import both
from PIL import Image


def modify_func(
    means3D: torch.Tensor,  # num_gauss x 3, means3D[:,1] = 0
    scales: torch.Tensor,   # num_gauss x 3, scales[:,1] = eps
    rotations: torch.Tensor,  # num_gauss x 4, 3D quaternions of 2D rotations
    time: float,
):
    return means3D, scales, rotations


def find_checkpoint(model_path: str, iteration: int):
    """
    Find a checkpoint .pth file inside `model_path`.

    If iteration >= 0, look for 'chkpnt{iteration}.pth'.
    If iteration < 0, pick the checkpoint with the largest iteration number.
    Returns full path or None if not found.
    """
    if iteration >= 0:
        ckpt_name = f"chkpnt{iteration}.pth"
        ckpt_path = os.path.join(model_path, ckpt_name)
        if os.path.exists(ckpt_path):
            return ckpt_path
        return None

    if not os.path.isdir(model_path):
        return None

    candidates = [
        f for f in os.listdir(model_path)
        if f.startswith("chkpnt") and f.endswith(".pth")
    ]
    if not candidates:
        return None

    best_iter = -1
    best_path = None
    for fname in candidates:
        # expect "chkpntXXXX.pth"
        num_str = fname[len("chkpnt"):-len(".pth")]
        try:
            it = int(num_str)
        except ValueError:
            continue
        if it > best_iter:
            best_iter = it
            best_path = os.path.join(model_path, fname)

    return best_path


def render_set(
    model_path,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    interp,
    extension,
    generate_points_path,
    mask_path,
    seg=False,
    head="img",
    pipeline_name="img",
    idx_offset: int = 0,
):
    if pipeline_name == "seg":
        render_dir = os.path.join(model_path, "render_mask")
    else:
        render_dir = os.path.join(model_path, "render_img")

    os.makedirs(render_dir, exist_ok=True)

    import gc

    for local_idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        idx = idx_offset + local_idx

        for i in range(interp):
            mask_means = None
            if mask_path != "-1":
                file_path = os.path.join(mask_path, "means3Dmasked", f"{idx:05d}.pt")
                if os.path.exists(file_path):
                    mm = torch.load(file_path, map_location="cpu")
                    mask_means = mm[:, [0, -1]]
                    del mm

            rendering = render(
                view,
                gaussians,
                pipeline,
                background,
                interp=interp,
                interp_idx=i,
                modify_func=modify_func,
                idx=idx,
                generate_points_path=generate_points_path,
                mask_means=mask_means,
                train=False,
                seg=seg,
                head=head,
            )["render"].detach().cpu()

            file_name = f"{idx:05d}_{i}{extension}"
            torchvision.utils.save_image(rendering, os.path.join(render_dir, file_name))

            del rendering, mask_means

        if (idx % 50) == 0:
            gc.collect()
            torch.cuda.empty_cache()


def _drop_camera_images(cam_list):
    for c in cam_list:
        for attr in [
            "original_image", "image", "gt_image",
            "original_mask", "mask", "alpha_mask",
            "depth", "depth_image"
        ]:
            if hasattr(c, attr):
                try:
                    setattr(c, attr, None)
                except Exception:
                    pass


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    interp: int,
    extension: str,
    generate_points_path,
    mask_path,
    pipeline_mode: str = "img",   # "img" | "seg" | "both"
    chunks: int = 1,
):
    with torch.no_grad():
        frames = len(os.listdir(f"{dataset.source_path}/original"))

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        ckpt_path = find_checkpoint(dataset.model_path, iteration)

        model_state = None
        loaded_iter = iteration

        if ckpt_path is not None:
            model_state, loaded_iter = torch.load(ckpt_path, map_location="cuda")


        dummy = gaussianModel["gs"](dataset.sh_degree, dataset.poly_degree, frames, use_dff=False)
        try:
            scene = Scene(dataset, dummy, load_iteration=loaded_iter, shuffle=False)
        except Exception:

            scene = Scene(dataset, dummy, load_iteration=None, shuffle=False)


        views_all = scene.getTestCameras()
        n_views = len(views_all)


        _drop_camera_images(views_all)
        if hasattr(scene, "train_cameras"):
            try:
                _drop_camera_images(scene.train_cameras)
            except Exception:
                pass


        if ckpt_path is not None:
            print(f"Loading checkpoint: {ckpt_path}")
            gaussians = gaussianModel["gs"](dataset.sh_degree, dataset.poly_degree, frames, use_dff=False)
            gaussians.restore(model_state, training_args=None, load_optimizer=False)
            has_seg = gaussians.has_seg_head() if hasattr(gaussians, "has_seg_head") else False
        else:
            print("No checkpoint found; using PLY/Scene gaussians.")
            gaussians = scene.gaussians
            has_seg = False

        def do_img(views_slice, idx_offset):
            render_set(
                dataset.model_path,
                loaded_iter,
                views_slice,
                gaussians,
                pipeline,
                background,
                interp,
                extension,
                generate_points_path,
                mask_path,
                seg=False,
                head="img",
                pipeline_name="img",
                idx_offset=idx_offset,
            )

        def do_seg(views_slice, idx_offset):
            head_name = "seg" if has_seg else "img"
            render_set(
                dataset.model_path,
                loaded_iter,
                views_slice,
                gaussians,
                pipeline,
                background,
                interp,
                extension,
                generate_points_path,
                mask_path,
                seg=True,
                head=head_name,
                pipeline_name="seg",
                idx_offset=idx_offset,
            )


        chunks = max(1, int(chunks))
        per = (n_views + chunks - 1) // chunks

        import gc
        for c in range(chunks):
            start = c * per
            if start >= n_views:
                break
            end = min(n_views, start + per)
            views_slice = views_all[start:end]

            print(f"[render] chunk {c+1}/{chunks}: views {start}..{end-1}")

            if pipeline_mode == "both":
                do_img(views_slice, start)
                do_seg(views_slice, start)
            elif pipeline_mode == "seg":
                do_seg(views_slice, start)
            else:
                do_img(views_slice, start)

            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--camera", type=str, default="mirror")
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--skip_train", action="store_false")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--poly_degree", type=int, default=1)
    parser.add_argument("--interp", type=int, default=1)
    parser.add_argument("--extension", type=str, default=".png")
    parser.add_argument("--mask_path", type=str, default="-1")
    parser.add_argument("--generate_points_path", type=str, default="-1")
    parser.add_argument("--pipeline", type=str, choices=["img", "seg", "both"], default="img")
    parser.add_argument("--chunks", type=int, default=1)


    args = get_combined_args(parser)
    model.gs_type = "gs"
    model.camera = args.camera
    model.distance = args.distance
    model.num_pts = args.num_pts
    model.poly_degree = args.poly_degree

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.interp,
        args.extension,
        generate_points_path=args.generate_points_path,
        mask_path=args.mask_path,
        pipeline_mode=args.pipeline,
        chunks=args.chunks
    )
