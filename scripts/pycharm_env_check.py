import os
import torch
import torchvision

from diff_gaussian_rasterization import GaussianRasterizer
from simple_knn._C import distCUDA2

print("cwd:", os.getcwd())
print("python env check OK")
print("torch:", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("torchvision:", torchvision.__version__)
print("device:", torch.cuda.get_device_name(0))
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("LD_LIBRARY_PATH starts:", os.environ.get("LD_LIBRARY_PATH", "")[:200])