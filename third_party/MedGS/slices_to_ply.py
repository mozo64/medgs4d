import os
import numpy as np
import cv2
import re
import glob
import nibabel as nib
import open3d as o3d
from skimage import measure
from scipy.ndimage import gaussian_filter
import argparse
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import threshold_otsu

def extract_numbers(filename):
    match = re.match(r'(\d+)_([\d]+)', filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)

def load_volume(render_folder, nifti_path=None):
    files = sorted(
        [f for f in os.listdir(render_folder) if f.endswith(".png") and not f.startswith('_')],
        key=extract_numbers
    )

    volume = []
    spacing = (1.0, 1.0, 1.0)

    if nifti_path:
        img = nib.load(nifti_path)
        spacing = img.header.get_zooms()[:3]
        print("Voxel spacing:", spacing)

    for frame in files:
        frame_path = os.path.join(render_folder, frame)
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        img = denoise_tv_chambolle(img.astype(float) / 255, weight=0.5)
        img = (img * 255.).astype(np.uint8)
        volume.append(img)

    volume = np.array(volume, dtype=np.float32)
    print("Volume shape (slices, height, width):", volume.shape)
    volume = gaussian_filter(volume, sigma=(6,1.3,1.3)) #due to high interpolation factor, we can use strong filtering without losing quality
    return volume, spacing


def main(input_folder, output_folder, nifti_folder, threshold=0, inter=8):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        print(filename)
        render_folder = os.path.join(input_folder, filename, "render")
        if not os.path.isdir(render_folder):
            continue
        #if not empyty render folder, continue
        if not os.listdir(render_folder):
            print(f"Skipping {render_folder} as it is empty.")
            continue

        # Only build a NIfTI path if a folder was provided
        nifti_path = None
        if nifti_folder:
            candidate = os.path.join(nifti_folder, f"{filename}.nii.gz")
            if os.path.exists(candidate):
                nifti_path = candidate
            else:
                print("No nifti file:", candidate, "— proceeding without NIfTI metadata.")

        volume, spacing = load_volume(render_folder, nifti_path)
        thresh = threshold if threshold > 0 else threshold_otsu(volume)

        verts, faces, normals, values = measure.marching_cubes(
            volume,
            level=thresh,
            # If no NIfTI, spacing remains default (1,1,1) set by load_volume
            spacing=[spacing[2] / inter, spacing[0], spacing[1]]
        )
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()

        out_path = os.path.join(output_folder, f"{filename}.ply")
        o3d.io.write_triangle_mesh(out_path, mesh, print_progress=True)
        print(f"Saved mesh to {out_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build 3D meshes from 2D slices")
    parser.add_argument("-i", "--input", required=True, help="Input folder")
    parser.add_argument("-o", "--output", required=True, help="Output folder")
    parser.add_argument("-n", "--nifti", default=None, help="Path to nifti files")
    parser.add_argument("--thresh", type=float, default=0, help="Marching cubes threshold")
    parser.add_argument("--inter", type=int, default=8, help="Interpolation factor")
    args = parser.parse_args()

    main(args.input, args.output, args.nifti, args.thresh, args.inter)
