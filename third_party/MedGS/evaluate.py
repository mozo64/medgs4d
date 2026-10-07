import os
import trimesh
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import nibabel as nib
from utils.mesh_utils import prepare_mesh, scale_mesh_to_match

def chamfer_distance(points_src, points_tgt, squared=False):
    tree_src = cKDTree(points_src)
    tree_tgt = cKDTree(points_tgt)
    dist_src_to_tgt, _ = tree_tgt.query(points_src)
    dist_tgt_to_src, _ = tree_src.query(points_tgt)
    if squared:
        dist_src_to_tgt = dist_src_to_tgt ** 2
        dist_tgt_to_src = dist_tgt_to_src ** 2
    return 0.5 * (np.mean(dist_src_to_tgt) + np.mean(dist_tgt_to_src))


def average_surface_distance(P, Q):
    return 0.5 * (np.mean(cKDTree(Q).query(P)[0]) + np.mean(cKDTree(P).query(Q)[0]))

def hausdorff_metrics(P, Q):
    dist_PQ = cKDTree(Q).query(P)[0]
    dist_QP = cKDTree(P).query(Q)[0]
    return max(np.max(dist_PQ), np.max(dist_QP)), np.percentile(np.concatenate([dist_PQ, dist_QP]), 95)

def evaluate_pair(pred_path, gt_path, nifti_path, mirror = False, edit = False):
    pred_mesh = trimesh.load(pred_path)
    gt_mesh = trimesh.load(gt_path)

    if mirror:
        pred_mesh.vertices[:, 0] *= -1


    nii = nib.load(nifti_path)
    affine = nii.affine

    if edit:
        resolution = 128
        voxel_origin = np.array([-1, -1, -1])
        voxel_size = 2.0 / (resolution - 1)
        verts_voxel = (pred_mesh.vertices - voxel_origin) / voxel_size
        verts_hom = np.c_[verts_voxel, np.ones(len(verts_voxel))]
        verts_mm = (affine @ verts_hom.T).T[:, :3]
        pred_mesh = trimesh.Trimesh(vertices=verts_mm, faces=pred_mesh.faces, process=False)

    #you have to experiment with settings here for the best mesh

    pred_mesh = scale_mesh_to_match(pred_mesh, gt_mesh)

    # pred_center = pred_mesh.vertices.mean(axis=0)
    # gt_center = gt_mesh.vertices.mean(axis=0)
    # pred_mesh.vertices += (gt_center - pred_center

    # pred_mesh_aligned = prepare_mesh(pred_mesh, gt_mesh, 90,180,90, 1)
    pred_mesh_aligned = prepare_mesh(pred_mesh, gt_mesh, 270,0,270, 100)
    # pred_mesh_aligned = prepare_mesh(pred_mesh, gt_mesh, 0, 0, 0, 1000)
    # pred_mesh_aligned = prepare_mesh(pred_mesh, gt_mesh, 0, 0, 180, 50)
    # pred_mesh_aligned = prepare_mesh(pred_mesh, gt_mesh, 0, 0, 180, 50)

    pred_pts = trimesh.sample.sample_surface(pred_mesh_aligned, 50000)[0]
    gt_pts = trimesh.sample.sample_surface(gt_mesh, 50000)[0]
    cd = chamfer_distance(pred_pts, gt_pts)
    asd = average_surface_distance(pred_pts, gt_pts)
    hd, hd95 = hausdorff_metrics(pred_pts, gt_pts)

    pred_mesh_aligned.visual.face_colors = [255, 0, 0, 70]
    gt_mesh.visual.face_colors = [0, 255, 0, 100]

    scene = trimesh.Scene([pred_mesh_aligned, gt_mesh])
    scene.show(flags={'wireframe': True})

    return {
        "ASD(mm)": round(asd, 4),
        "CD(mm)": round(cd, 4),
        "HD(mm)": round(hd, 4),
        "HD95(mm)": round(hd95, 4),
    }


def evaluate_all(pred_dir, gt_dir, nifti_dir, output_csv="results.csv"):
    all_results = []
    for file in sorted(os.listdir(pred_dir)):
        if not file.endswith("70_pred.ply"):
            continue
        name = file.replace("_pred.ply", "")
        pred_path = os.path.join(pred_dir, file)
        gt_path = os.path.join(gt_dir, fr"{name}_gt.ply")
        nifti_path = os.path.join(nifti_dir, fr"{name}.nii.gz")
        if not os.path.exists(gt_path):
            print(f"Missing GT for {name}, skipping.")
            continue
        print(f"Evaluating {name}...")
        metrics = evaluate_pair(pred_path, gt_path, nifti_path, True, False)
        metrics["Case"] = name
        all_results.append(metrics)

    df = pd.DataFrame(all_results)
    df = df[["Case", "ASD(mm)", "CD(mm)", "HD(mm)", "HD95(mm)"]]#, "DSC", "IoU"]]
    avg = df.drop(columns="Case").mean().to_dict()
    avg["Case"] = "Average"
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    df.to_csv(output_csv, index=False)
    print("Saved results to", output_csv)


if __name__ == '__main__':
    evaluate_all('prostate/prostate_funsr_predictions', 'prostate/gt_slicer_output', 'prostate/prostate_dset/val/us_labels')
