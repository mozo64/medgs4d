"""Isolated GPU worker: endpoint-only preparation/training, then frozen rendering."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("config", type=Path)
parser.add_argument("mode", choices=["prepare", "canonical", "train", "render", "selftest"])
parser.add_argument("--case")
parser.add_argument("--method")
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()
E = json.loads(args.config.read_text())
ROOT = Path(E["root"])
SOURCE = ROOT / "sources"
sys.path.insert(0, str(SOURCE / "medgs4d"))
import numpy as np
import pandas as pd
import torch
from medgs4d.config import (CanonicalConfig, DeformationConfig, TrainingConfig,
    MedGS4DConfig, SplitConfig, load_medgs4d_config, validate_medgs4d_config)
from medgs4d.data import StudyManifest, save_study_manifest, load_study_manifest
from medgs4d.canonical import (train_canonical_model, load_frozen_canonical,
    get_camera_for_slice, build_canonical_paths)
from medgs4d.runs import build_run_paths, write_json, find_latest_checkpoint
from medgs4d import training as tr
from publication_methods import field_class, install_training_variant

random.seed(E["seed"])
np.random.seed(E["seed"])
torch.manual_seed(E["seed"])
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# Custom CUDA rasterization can still be nondeterministic; the seed is recorded.

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def observed(case, phase):
    assert phase in (0, 50), "Training worker cannot load intermediate CT."
    import nibabel as nib
    p = Path(E["data_root"]) / case / f"ct_{case}_frame{phase // 10}.nii.gz"
    image = nib.load(str(p))
    volume = image.get_fdata(dtype=np.float32).transpose(2, 1, 0).copy()
    assert volume.shape == (128, 128, 128)
    assert np.isfinite(volume).all()
    assert volume.min() >= -1e-6 and volume.max() <= 1.0 + 1e-6
    return p, volume, image


def prepare(case):
    dest = ROOT / "prepared" / case
    rows, paths, hashes, headers = [], {}, {}, {}
    dest.mkdir(parents=True, exist_ok=True)
    for phase in (0, 50):
        source, volume, image = observed(case, phase)
        hashes[str(phase)] = sha(source)
        headers[str(phase)] = {"affine": image.affine.tolist(), "shape": list(image.shape)}
        p = dest / f"phase_{phase:02d}.npy"
        if p.exists():
            assert np.array_equal(np.load(p), volume), f"Prepared data changed: {p}"
        else:
            np.save(p, volume)
        paths[float(phase)] = str(p)
        rows += [{"PhasePercent": phase, "SliceIndex": k, "SliceCoordinate": float(k)}
                 for k in range(128)]
    assert np.allclose(headers["0"]["affine"], headers["50"]["affine"])
    fingerprint = dest / "observed_sources.json"
    record = {"sha256": hashes, "headers": headers, "transpose": [2, 1, 0]}
    if fingerprint.exists():
        assert json.loads(fingerprint.read_text()) == record, "Observed source changed."
    write_json(fingerprint, record)
    csv = dest / "phase_slice_manifest.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    summary = dest / "phase_summary.csv"
    pd.DataFrame({"PhasePercent": [0, 50], "SliceCount": [128, 128]}).to_csv(summary, index=False)
    study = StudyManifest(
        study_name=f"benchmark_{case}", patient_id=case.split("_")[0],
        study_instance_uid=f"benchmark_{case}", phases=(0.0, 50.0),
        slice_count=128, volume_shape=(128, 128, 128), hu_window=(0.0, 1.0),
        denoise_sigma=(0.0, 0.0, 0.0), raw_volume_paths=paths,
        denoised_volume_paths=paths.copy(), phase_slice_manifest_path=str(csv),
        phase_summary_path=str(summary),
    )
    save_study_manifest(study, dest)
    print("Prepared endpoints:", case, flush=True)


def canonical_paths(case):
    return build_canonical_paths(ROOT / "canonical", f"benchmark_{case}",
                                 f"phase00_iter{E['canonical_iterations']}")


def canonical_checkpoint(case):
    paths = canonical_paths(case)
    return paths.model / f"chkpnt{E['canonical_iterations']}.pth"


def train_canonical(case):
    paths = canonical_paths(case)
    checkpoint = canonical_checkpoint(case)
    if paths.metadata.exists() and checkpoint.exists():
        print("Canonical complete:", case, flush=True)
        return
    study = load_study_manifest(ROOT / "prepared" / case)
    resume = paths.root.exists()
    if resume:
        assert args.resume, "Existing canonical run: enable RESUME for this experiment."
        if not list(paths.model.glob("chkpnt*.pth")):
            raise RuntimeError(f"Interrupted before first canonical checkpoint: choose a new experiment ID or inspect {paths.root}.")
    config = CanonicalConfig(study_name=study.study_name, run_name=paths.root.name,
        canonical_phase=0.0, iterations=E["canonical_iterations"], seed=E["seed"],
        poly_degree=2, batch_size=3, camera="mirror", representation="raw")
    start = time.perf_counter()
    train_canonical_model(study, SOURCE / "MedGS", ROOT / "canonical", config,
                          resume=resume, log_every=100)
    write_json(paths.root / "resource_segment.json", {
        "wall_seconds": time.perf_counter() - start, "resumed": resume,
        "peak_gpu_gb": None, "note": "Canonical runs in a nested process; allocator peak is not captured.",
        "checkpoint_sha256": sha(checkpoint),
    })


def dynamic_paths(case, method):
    return build_run_paths(ROOT / "methods", f"benchmark_{case}", method)


def config_for(case, method):
    kw = dict(iterations=E["temporal_iterations"], learning_rate=5e-4,
        checkpoint_every=500, log_every=25, validate_every=0, validation_samples=0,
        seed=E["seed"], phase_jitter_initial_std=0.0)
    if method == "linear_pseudo":
        kw["pseudo_weight"] = 0.1
    if method == "cycle_v2":
        kw.update(pairwise_cycle_weight=0.05, pairwise_agreement_weight=0.01,
                  pairwise_transport_weight=0.001)
    if method in {"consistent_cycle", "trajectory_cycle"}:
        # This flag allocates the auxiliary network. Actual loss weights are in new_options.
        kw["pairwise_cycle_weight"] = E["new_options"]["endpoint_image_weight"]
    return MedGS4DConfig(study_name=f"benchmark_{case}", run_name=method,
        data_dir=str(ROOT / "prepared" / case), canonical_model_dir=str(canonical_paths(case).root),
        medgs_repository=str(SOURCE / "MedGS"), canonical_phase=0.0,
        split=SplitConfig(mode="full"), deformation=DeformationConfig(chunk_size=32768),
        training=TrainingConfig(**kw), target_representation="raw",
        canonical_checkpoint=str(canonical_checkpoint(case)),
        canonical_checkpoint_iteration=E["canonical_iterations"])


def train(case, method):
    paths = dynamic_paths(case, method)
    expected = paths.checkpoints / f"deformation_iter_{E['temporal_iterations']:06d}.pth"
    if paths.completion.exists() and expected.exists():
        done = json.loads(paths.completion.read_text())
        assert done["iteration"] == E["temporal_iterations"]
        print("Temporal complete:", case, method, flush=True)
        return
    config = config_for(case, method)
    validate_medgs4d_config(config)
    study = load_study_manifest(Path(config.data_dir))
    assert set(study.phases) == {0.0, 50.0}
    canonical = load_frozen_canonical(Path(config.canonical_model_dir), SOURCE / "MedGS",
                                      checkpoint=canonical_checkpoint(case), device="cuda:0")
    checkpoint = find_latest_checkpoint(paths.checkpoints)
    resume = checkpoint is not None
    if paths.root.exists():
        assert args.resume, "Existing temporal run: enable RESUME."
    paths.root.mkdir(parents=True, exist_ok=True)
    write_json(paths.root / "method.json", {
        "method": method, "new_options": E["new_options"],
        "canonical_sha256": sha(canonical_checkpoint(case)),
        "architecture": field_class(method).__name__,
    })
    install_training_variant(method, E["new_options"])
    torch.cuda.reset_peak_memory_stats()
    tr.train_medgs4d(study, canonical, paths, config, resume=resume,
                    resume_checkpoint=checkpoint, final_evaluation=False)
    assert expected.exists()


def render(case, method):
    paths = dynamic_paths(case, method)
    config = load_medgs4d_config(paths.config)
    checkpoint = paths.checkpoints / f"deformation_iter_{E['temporal_iterations']:06d}.pth"
    assert paths.completion.exists() and checkpoint.exists()
    dest = ROOT / "predictions" / case / method
    dest.mkdir(parents=True, exist_ok=True)
    hashes = {"temporal": sha(checkpoint), "canonical": sha(canonical_checkpoint(case))}
    provenance = dest / "render.json"
    if provenance.exists():
        saved = json.loads(provenance.read_text())
        assert saved["checkpoint_sha256"] == hashes
        if all((dest / f"phase_{p:02d}.npy").exists() for p in E["evaluation_phases"]):
            for p in E["evaluation_phases"]:
                assert sha(dest / f"phase_{p:02d}.npy") == saved["prediction_sha256"][str(p)]
            print("Predictions complete:", case, method, flush=True)
            return
    canonical = load_frozen_canonical(Path(config.canonical_model_dir), SOURCE / "MedGS",
                                      checkpoint=canonical_checkpoint(case), device="cuda:0")
    field = field_class(method)(canonical, config.deformation, 0.0, seed=E["seed"])
    actual = tr.load_checkpoint(checkpoint, field, device="cuda:0", config=config)
    assert actual == E["temporal_iterations"]
    field.model.eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    clipped, pred_hashes = {}, {}
    with torch.no_grad():
        for phase in E["evaluation_phases"]:
            view, _ = field.build_phase_state(phase / 100.0, use_checkpointing=False)
            slices, count, outside = [], 0, 0
            for k in range(128):
                image = canonical.runtime.render(get_camera_for_slice(canonical, k), view,
                    canonical.pipeline, canonical.background)["render"]
                outside += int(((image < 0) | (image > 1)).sum())
                count += image.numel()
                slices.append(image.clamp(0, 1).mean(dim=0).cpu().numpy().astype(np.float32))
            volume = np.stack(slices)
            assert volume.shape == (128, 128, 128) and np.isfinite(volume).all()
            p = dest / f"phase_{phase:02d}.npy"
            np.save(p, volume)
            pred_hashes[str(phase)] = sha(p)
            clipped[str(phase)] = outside / count
            print(case, method, f"{phase}% rendered", flush=True)
    torch.cuda.synchronize()
    write_json(provenance, {
        "checkpoint_sha256": hashes, "prediction_sha256": pred_hashes,
        "phases": E["evaluation_phases"], "shape_order": "ZYX",
        "prediction_transform": "clamp RGB to [0,1], then mean channels",
        "rgb_clipped_fraction": clipped, "seconds": time.perf_counter() - start,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "primary_parameters": field.parameter_count,
    })


def selftest():
    from types import SimpleNamespace
    from medgs4d.cycle import PairwiseTransport
    from publication_methods import EndpointTrajectoryField, consistency_losses
    xyz = torch.randn(31, 3)
    logits = torch.randn(31, 1)
    canonical = SimpleNamespace(xyz=xyz, xz=xyz[:, [0, 2]], m_logits=logits,
                                m=logits.sigmoid(), gaussians=SimpleNamespace())
    field = EndpointTrajectoryField(canonical, DeformationConfig(hidden_dim=16,
        hidden_layers=2, chunk_size=11), 0.0, seed=42)
    with torch.no_grad():
        for name, p in field.model.named_parameters():
            if "endpoint" in name or "residual.2" in name:
                p.normal_(0, 0.02)
    q0 = field.build_phase_state(0.0)[1]
    assert torch.allclose(q0["relative_deformation"], torch.zeros(31, 3))
    features = field.model.backbone(field.spatial_features)
    q1 = field.build_phase_state(0.5)[1]
    assert torch.allclose(q1["relative_deformation"], field.model.endpoint(features), atol=1e-6)
    assert torch.equal(q1["dynamic_xyz"][:, 1], xyz[:, 1])
    pair = PairwiseTransport(field, hidden_dim=16, hidden_layers=2, time_frequencies=2)
    same = pair.transport(q1, source_time=1.0, target_time=1.0)
    assert torch.allclose(same["dynamic_xyz"], q1["dynamic_xyz"])
    with torch.no_grad():
        for p in pair.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    losses = consistency_losses(field, pair, torch.arange(17), 0.31, 0.72, True)
    sum(losses).backward()
    for model in [field.model, pair]:
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
    # Round-trip through the same strict state-dict format used by training/inference.
    import io
    buffer = io.BytesIO()
    torch.save(field.model.state_dict(), buffer)
    buffer.seek(0)
    restored = EndpointTrajectoryField(canonical, field.config, 0.0, seed=42)
    restored.model.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    assert torch.allclose(restored.predict_relative_deformation(0.2),
                          field.predict_relative_deformation(0.2))
    print("CPU model tests passed: endpoints, fixed y, identity, gradients, checkpoint.", flush=True)


if args.mode == "selftest":
    selftest()
else:
    assert args.case in E["test_cases"]
    if args.mode == "prepare":
        prepare(args.case)
    else:
        assert torch.cuda.is_available(), "A working WORF CUDA environment is required."
        torch.cuda.set_device(0)
        if args.mode == "canonical":
            train_canonical(args.case)
        else:
            assert args.method in E["methods"]
            {"train": train, "render": render}[args.mode](args.case, args.method)
