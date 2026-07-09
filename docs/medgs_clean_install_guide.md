# Clean MedGS Setup on Ubuntu SSH Server

This note documents the clean setup we used to install and smoke-test the GMUM MedGS repository on an Ubuntu server accessed from a terminal/Termux-style SSH session.

The main practical lesson: do **not** install heavy CUDA/PyTorch packages in `$HOME` when `$HOME` lives on a small shared root filesystem. Use a large SSD-backed workspace instead.

## Target machine

Observed server characteristics:

- OS context: Ubuntu server, used over SSH.
- GPU: NVIDIA RTX A5500.
- Driver reported CUDA runtime compatibility around CUDA 12.4.
- System `/home` was on the root filesystem and became nearly full.
- Fast workspace: `/opt/jupyterhub/fast`, about **3.4 TB total** with roughly **0.9 TB free** during setup.
- Slow workspace: `/opt/jupyterhub/slow`, much larger but HDD/slow storage.
- Chosen workspace: `/opt/jupyterhub/fast/mtm_medgs_stack`.

The exact path is user/server-specific. In general, replace it with a writable SSD-backed directory with enough free space.

---

## 0. Use a persistent SSH session

When connecting from Termux or any unstable SSH client, run long commands inside `tmux`. Otherwise training may stop if the SSH connection drops.

Start a persistent session:

```bash
tmux new -s medgs
```

Detach without stopping the job:

```text
Ctrl-b, then d
```

Reattach later:

```bash
tmux attach -t medgs
```

List sessions:

```bash
tmux ls
```

Kill the session when done:

```bash
tmux kill-session -t medgs
```

Recommended workflow from Termux/local terminal:

```bash
ssh mtm@<server>
tmux new -s medgs
```

After reconnecting:

```bash
ssh mtm@<server>
tmux attach -t medgs
```

---

## 1. Check storage before installing

```bash
df -hT
df -hT /home /opt/jupyterhub/fast /opt/jupyterhub/slow
```

Why: PyTorch/CUDA wheels and Conda packages are large. Installing them into a small `$HOME` can fill the root filesystem.

---

## 2. Create a clean workspace on the fast disk

If the user does not own the target directory, create and assign only that directory:

```bash
export MEDGS_ROOT=/opt/jupyterhub/fast/mtm_medgs_stack

sudo mkdir -p "$MEDGS_ROOT"/{src,cache/pip,cache/conda_pkgs,envs}
sudo chown -R "$USER:$USER" "$MEDGS_ROOT"

export PIP_CACHE_DIR="$MEDGS_ROOT/cache/pip"
export CONDA_PKGS_DIRS="$MEDGS_ROOT/cache/conda_pkgs"

df -hT "$MEDGS_ROOT"
touch "$MEDGS_ROOT/write_test" && rm "$MEDGS_ROOT/write_test" && echo "write OK"
```

Why: this keeps the repository, Conda environments, package cache, pip cache, and temporary files away from `/home`.

---

## 3. Install Miniforge into the fast workspace

```bash
export MEDGS_ROOT=/opt/jupyterhub/fast/mtm_medgs_stack
export PIP_CACHE_DIR="$MEDGS_ROOT/cache/pip"
export CONDA_PKGS_DIRS="$MEDGS_ROOT/cache/conda_pkgs"

cd "$MEDGS_ROOT"

wget -O Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

bash Miniforge3-Linux-x86_64.sh -b -p "$MEDGS_ROOT/miniforge3"

source "$MEDGS_ROOT/miniforge3/bin/activate"

which conda
which mamba
conda --version
mamba --version
```

Why: we install a local Miniforge under the project workspace, not in `$HOME`.

---

## 4. Configure Conda package and environment locations

```bash
export MEDGS_ROOT=/opt/jupyterhub/fast/mtm_medgs_stack
source "$MEDGS_ROOT/miniforge3/bin/activate"

conda config --env --set auto_activate_base false
conda config --env --add envs_dirs "$MEDGS_ROOT/envs"
conda config --env --add pkgs_dirs "$MEDGS_ROOT/cache/conda_pkgs"

conda config --show envs_dirs
conda config --show pkgs_dirs
```

Why: package downloads and environments should stay on the fast disk.

---

## 5. Create the MedGS Python/CUDA environment

```bash
export MEDGS_ROOT=/opt/jupyterhub/fast/mtm_medgs_stack
export CONDA_PKGS_DIRS="$MEDGS_ROOT/cache/conda_pkgs"
export PIP_CACHE_DIR="$MEDGS_ROOT/cache/pip"

source "$MEDGS_ROOT/miniforge3/bin/activate"

mamba create -p "$MEDGS_ROOT/envs/medgs38" \
  python=3.8 \
  cuda-toolkit=12.4 \
  ninja setuptools wheel pip \
  -c conda-forge -c nvidia \
  -y

conda activate "$MEDGS_ROOT/envs/medgs38"

which python
python --version
which nvcc
nvcc --version
```

Expected:

```text
Python 3.8.x
nvcc: Cuda compilation tools, release 12.4
```

Why: MedGS expects Python 3.8 and CUDA Toolkit 12.x for compiling PyTorch/CUDA extensions.

---

## 6. Add a robust activation script

Create environment-specific activation/deactivation hooks:

```bash
export MEDGS_ROOT=/opt/jupyterhub/fast/mtm_medgs_stack
source "$MEDGS_ROOT/miniforge3/bin/activate"
conda activate "$MEDGS_ROOT/envs/medgs38"

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/medgs_env.sh" <<'SH'
export MEDGS_ROOT=/opt/jupyterhub/fast/mtm_medgs_stack
export PIP_CACHE_DIR="$MEDGS_ROOT/cache/pip"
export CONDA_PKGS_DIRS="$MEDGS_ROOT/cache/conda_pkgs"
export TMPDIR="$MEDGS_ROOT/tmp"

mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$TMPDIR"

export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"

export CUDA_INC_DIR="$CONDA_PREFIX/targets/x86_64-linux/include"
export CUDA_LIB_DIR="$CONDA_PREFIX/targets/x86_64-linux/lib"

export CPATH="$CUDA_INC_DIR${CPATH:+:$CPATH}"
export C_INCLUDE_PATH="$CUDA_INC_DIR${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$CUDA_INC_DIR${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$CUDA_LIB_DIR${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_LIB_DIR:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=4

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
SH

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/medgs_env.sh" <<'SH'
unset CUDA_VISIBLE_DEVICES
unset CUDA_HOME
unset CUDA_PATH
unset CUDA_INC_DIR
unset CUDA_LIB_DIR
unset TORCH_CUDA_ARCH_LIST
unset MAX_JOBS
unset CC
unset CXX
unset CUDAHOSTCXX
SH
```

Then verify:

```bash
conda deactivate
conda activate "$MEDGS_ROOT/envs/medgs38"

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "TMPDIR=$TMPDIR"
echo "CC=$CC"
echo "CXX=$CXX"
which nvcc
nvcc --version
```

Why:

- `CUDA_HOME` points PyTorch extension builds to the Conda CUDA Toolkit.
- `CUDA_INC_DIR`/`CUDA_LIB_DIR` expose CUDA headers and libraries.
- `CC`, `CXX`, and `CUDAHOSTCXX` force system GCC/G++ instead of Conda GCC 14, which CUDA 12.4 may reject.
- `TORCH_CUDA_ARCH_LIST=8.6` targets RTX A5500/Ampere.

---

## 7. Install compatible PyTorch and TorchVision

```bash
python -m pip install --upgrade pip

pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python - <<'PY'
import torch
import torchvision
print("torch:", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("torchvision:", torchvision.__version__)
from torchvision.utils import save_image
print("torchvision import OK")
PY
```

Expected:

```text
torch: 2.4.0+cu121 12.1 True
torchvision: 0.19.0+cu121
torchvision import OK
```

Why: installing both from the same PyTorch wheel source avoids the `RuntimeError: operator torchvision::nms does not exist` issue caused by mixing Conda-Forge PyTorch with PyTorch-channel/Pip TorchVision.

---

## 8. Clone MedGS with submodules

```bash
export MEDGS_ROOT=/opt/jupyterhub/fast/mtm_medgs_stack
cd "$MEDGS_ROOT/src"

git clone --recursive https://github.com/gmum/MedGS.git

cd "$MEDGS_ROOT/src/MedGS"
git submodule update --init --recursive

ls submodules
ls submodules/diff-gaussian-rasterization
ls submodules/simple-knn
```

Expected submodules include:

```text
diff-gaussian-rasterization
fused-ssim
simple-knn
```

---

## 9. Build MedGS CUDA submodules

```bash
cd /opt/jupyterhub/fast/mtm_medgs_stack/src/MedGS

rm -rf submodules/diff-gaussian-rasterization/build
rm -rf submodules/simple-knn/build

pip install --no-build-isolation -v submodules/diff-gaussian-rasterization
pip install --no-build-isolation -v submodules/simple-knn
```

Verify:

```bash
python - <<'PY'
import torch
from diff_gaussian_rasterization import GaussianRasterizer
from simple_knn._C import distCUDA2

print("torch:", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("MedGS CUDA imports OK")
PY
```

Expected:

```text
torch: 2.4.0+cu121 12.1 True
device: NVIDIA RTX A5500
MedGS CUDA imports OK
```

Why: these are custom CUDA extensions required by MedGS/3D Gaussian Splatting.

---

## 10. Install the remaining Python dependencies

Do **not** run `pip install -r requirements.txt` blindly, because `requirements.txt` contains `torch` and `torchvision`, which may replace the working versions.

Install the remaining packages explicitly:

```bash
pip install \
  opencv-python \
  tqdm \
  matplotlib \
  trimesh \
  plyfile \
  nibabel \
  open3d \
  scikit-image
```

Verify:

```bash
python - <<'PY'
import torch
import torchvision
import cv2
import tqdm
import matplotlib
import trimesh
import plyfile
import nibabel
import open3d
import skimage
from diff_gaussian_rasterization import GaussianRasterizer
from simple_knn._C import distCUDA2

print("all imports OK")
print("torch:", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("torchvision:", torchvision.__version__)
PY
```

---

## 11. Smoke-test training

MedGS currently includes a small prostate example dataset:

```bash
cd /opt/jupyterhub/fast/mtm_medgs_stack/src/MedGS

DATASET="$PWD/data/prostate"

ls "$DATASET"
ls "$DATASET/original" | head
```

Run a short training job:

```bash
rm -rf output/smoke_img
mkdir -p output

python train.py \
  -s "$DATASET" \
  -m output/smoke_img \
  --iterations 1000 \
  --save_iterations 1000 \
  --test_iterations 1000 \
  --checkpoint_iterations 1000 \
  --poly_degree 2
```

Expected outputs include:

```text
Training complete.
output/smoke_img/point_cloud/iteration_1000/point_cloud.ply
output/smoke_img/plots/img_1000.png
output/smoke_img/chkpnt1000.pth
```

---

## 12. Smoke-test rendering

```bash
python render.py \
  --model_path output/smoke_img \
  --iteration -1 \
  --pipeline img

find output/smoke_img -maxdepth 5 -type f | grep render | head
```

Why: this confirms that `torchvision.utils.save_image` works and the earlier `torchvision::nms` issue is fixed.

---

## 13. Preview rendered images in an SSH terminal

When working over SSH, especially from Termux or another terminal-only client, it is useful to preview the rendered PNG files directly in the terminal. The following script selects the rendered frame with the strongest visible signal, crops the dark background, increases contrast, and displays it using ANSI RGB block characters.

```bash
cd /opt/jupyterhub/fast/mtm_medgs_stack/src/MedGS

python - <<'PY'
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import shutil

render_dir = Path("output/smoke_img/render_img")
files = sorted(render_dir.glob("*.png"))

if not files:
    raise SystemExit("No render PNG files found.")

best = None
best_score = -1

for path in files:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)

    p99 = np.percentile(arr, 99)
    score = float(arr[arr > max(2, p99 * 0.25)].sum()) if p99 > 0 else 0.0

    if score > best_score:
        best_score = score
        best = path

img = Image.open(best).convert("L")
arr = np.asarray(img)

threshold = max(2, int(np.percentile(arr, 99) * 0.20))
mask = arr > threshold

if mask.any():
    ys, xs = np.where(mask)
    pad = 12
    x0 = max(xs.min() - pad, 0)
    x1 = min(xs.max() + pad + 1, img.width)
    y0 = max(ys.min() - pad, 0)
    y1 = min(ys.max() + pad + 1, img.height)
    img = img.crop((x0, y0, x1, y1))

img = ImageOps.autocontrast(img, cutoff=0)
img = ImageOps.equalize(img)
img = img.convert("RGB")

term = shutil.get_terminal_size((160, 80))
cols = min(term.columns, 180)
rows = min(max(20, term.lines - 4), 80)

img.thumbnail((cols, rows * 2), Image.Resampling.LANCZOS)

w, h = img.size
if h < rows * 2:
    scale = min(cols / w, (rows * 2) / h)
    if scale > 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.NEAREST)

w, h = img.size
if h % 2 == 1:
    img = img.crop((0, 0, w, h - 1))
    w, h = img.size

print(f"Best frame: {best}")
print(f"Terminal preview size: {w}x{h} image pixels, {w}x{h//2} terminal chars")

px = img.load()
for y in range(0, h, 2):
    line = []
    for x in range(w):
        r1, g1, b1 = px[x, y]
        r2, g2, b2 = px[x, y + 1]
        line.append(
            f"\\033[38;2;{r1};{g1};{b1}m"
            f"\\033[48;2;{r2};{g2};{b2}m"
            "▀"
        )
    print("".join(line) + "\\033[0m")
PY
```

Why: this gives a quick qualitative check of the render without copying files from the server or using a graphical desktop session. It is only a terminal preview; for proper inspection, copy the PNG files or a generated contact sheet to a local machine.

---

## 14. Daily usage

For future sessions:

```bash
export MEDGS_ROOT=/opt/jupyterhub/fast/mtm_medgs_stack
source "$MEDGS_ROOT/miniforge3/bin/activate"
conda activate "$MEDGS_ROOT/envs/medgs38"

cd "$MEDGS_ROOT/src/MedGS"
```

Quick health check:

```bash
python - <<'PY'
import torch
import torchvision
from diff_gaussian_rasterization import GaussianRasterizer
from simple_knn._C import distCUDA2
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torchvision.__version__)
print(torch.cuda.get_device_name(0))
print("MedGS environment OK")
PY
```

---

## Notes and pitfalls

- Avoid installing large CUDA/PyTorch packages into `$HOME`.
- Keep Conda package cache, pip cache, and temp files on the fast disk.
- Do not mix Conda-Forge PyTorch with PyTorch-channel or pip TorchVision.
- Do not run `pip install -r requirements.txt` directly unless `torch` and `torchvision` are removed or pinned safely.
- Use `--no-build-isolation` when building local CUDA submodules so the build sees the active environment.
- Force `/usr/bin/gcc` and `/usr/bin/g++` if Conda GCC is too new for CUDA.
- For RTX A5500, `TORCH_CUDA_ARCH_LIST=8.6` is appropriate.
