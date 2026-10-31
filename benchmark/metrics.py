"""Original benchmark full-volume metrics (unchanged definitions)."""
import math
import numpy as np
import torch
import torch.nn.functional as F

def gaussian_1d(window_size=11, sigma=1.5):
    values = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)])
    return values / values.sum()


def ssim3d(original, generated, window_size=11):
    one_d = gaussian_1d(window_size).to(original.device, original.dtype)
    two_d = one_d[:, None] * one_d[None, :]
    window = one_d[:, None, None] * two_d[None, :, :]
    window = window[None, None, ...]
    padding = window_size // 2
    mu1 = F.conv3d(original, window, padding=padding)
    mu2 = F.conv3d(generated, window, padding=padding)
    mu1_sq = mu1.square()
    mu2_sq = mu2.square()
    mu12 = mu1 * mu2
    sigma1_sq = F.conv3d(original.square(), window, padding=padding) - mu1_sq
    sigma2_sq = F.conv3d(generated.square(), window, padding=padding) - mu2_sq
    sigma12 = F.conv3d(original * generated, window, padding=padding) - mu12
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    score = (2 * mu12 + c1) * (2 * sigma12 + c2) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return score.mean()


def ncc3d(original, generated, window_size=9):
    window = torch.ones(1, 1, window_size, window_size, window_size, device=original.device, dtype=original.dtype)
    padding = window_size // 2
    window_volume = window_size ** 3
    original_sum = F.conv3d(original, window, padding=padding)
    generated_sum = F.conv3d(generated, window, padding=padding)
    original_sq_sum = F.conv3d(original.square(), window, padding=padding)
    generated_sq_sum = F.conv3d(generated.square(), window, padding=padding)
    cross_sum = F.conv3d(original * generated, window, padding=padding)
    mean_original = original_sum / window_volume
    mean_generated = generated_sum / window_volume
    cross = cross_sum - mean_generated * original_sum - mean_original * generated_sum + mean_original * mean_generated * window_volume
    var_original = original_sq_sum - 2 * mean_original * original_sum + mean_original.square() * window_volume
    var_generated = generated_sq_sum - 2 * mean_generated * generated_sum + mean_generated.square() * window_volume
    return (cross.square() / (var_original * var_generated + 1e-05)).mean()
@torch.no_grad()
def evaluate_volume(gt, prediction, device="cpu"):
    assert gt.shape == prediction.shape == (128, 128, 128)
    assert np.isfinite(gt).all() and np.isfinite(prediction).all()
    a = torch.as_tensor(np.ascontiguousarray(gt), dtype=torch.float32, device=device)[None, None]
    b = torch.as_tensor(np.ascontiguousarray(prediction), dtype=torch.float32, device=device)[None, None]
    mse = ((a - b) ** 2).mean()
    energy = a.square().mean()
    assert energy > 0
    return {"PSNR": float(-10 * torch.log10(mse)), "NCC": float(ncc3d(a, b)),
            "SSIM": float(ssim3d(a, b)), "NMSE": float(100 * mse / energy)}
