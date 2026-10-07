"""Temporal Gaussian models and endpoint-only consistency objectives."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from medgs4d.deformation import DeformationField
from medgs4d.cycle import PairwiseTransport, state_to_view
from medgs4d import training as tr


class EndpointTrajectoryMLP(torch.nn.Module):
    """Shared spatial backbone with endpoint and time-conditioned residual heads."""
    def __init__(self, spatial_dim, hidden_dim, hidden_layers):
        super().__init__()
        layers, dim = [], spatial_dim
        for _ in range(hidden_layers):
            layers += [torch.nn.Linear(dim, hidden_dim), torch.nn.SiLU()]
            dim = hidden_dim
        self.backbone = torch.nn.Sequential(*layers)
        self.endpoint = torch.nn.Linear(hidden_dim, 3)
        self.residual = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + 3, hidden_dim), torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 3),
        )
        for head in [self.endpoint, self.residual[-1]]:
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)

    def forward(self, spatial, u):
        features = self.backbone(spatial)
        time = features.new_tensor([u, u * u, u * (1.0 - u)])
        time = time.unsqueeze(0).expand(len(features), -1)
        endpoint = self.endpoint(features)
        residual = self.residual(torch.cat([features, time], dim=1))
        return u * endpoint + u * (1.0 - u) * residual


class EndpointTrajectoryField(DeformationField):
    """Use local u = phase / 50 for the observed half-cycle."""
    def __init__(self, canonical, config, canonical_phase, seed=42):
        super().__init__(canonical, config, canonical_phase, seed=seed)
        assert float(canonical_phase) == 0.0
        torch.manual_seed(seed)
        self.model = EndpointTrajectoryMLP(
            self.spatial_features.shape[1], config.hidden_dim, config.hidden_layers
        ).to(device=self.xyz.device, dtype=self.xyz.dtype)

    def predict_relative_deformation(self, respiratory_time, *,
                                    spatial_features=None, use_checkpointing=True):
        u = float(respiratory_time) / 0.5
        assert -1e-7 <= u <= 1.0 + 1e-7, "Trajectory supports phases 0–50%."
        features = self.spatial_features if spatial_features is None else spatial_features
        parts = []
        for start in range(0, len(features), self.config.chunk_size):
            selected = features[start:start + self.config.chunk_size]
            if use_checkpointing and torch.is_grad_enabled():
                value = checkpoint(self.model, selected, u, use_reentrant=False)
            else:
                value = self.model(selected, u)
            parts.append(value)
        return torch.cat(parts)

    def normalization_dict(self):
        result = super().normalization_dict()
        result.update(architecture="endpoint_residual", local_time="phase_percent / 50")
        return result


def detach_state(state):
    return {key: value.detach() for key, value in state.items()}


def geometric_subset(field, indices):
    """Retain global normalization while selecting Gaussian identities."""
    subset = copy.copy(field)
    for name in ["xyz", "xz", "m", "m_logits", "spatial_features", "normalized_coordinates"]:
        setattr(subset, name, getattr(field, name).index_select(0, indices))
    return subset


def transport_subset(transport, field):
    # A shallow copy shares learned parameters; only its geometry reference changes.
    result = copy.copy(transport)
    result._modules = dict(transport._modules)
    result.field = field
    return result


def primary_state(field, u):
    return field.build_phase_state(0.5 * float(u), use_checkpointing=True)[1]


def distance(transport, first, second):
    # x,z displacement and bounded m are normalized by canonical scales.
    return transport.geometry_distance(first, second)


def paired_alignment(transport, prediction, target, joint):
    loss = distance(transport, prediction, detach_state(target))
    if joint:
        loss = 0.5 * (loss + distance(transport, detach_state(prediction), target))
    return loss


def consistency_losses(field, transport, indices, u, v, joint):
    """Return losses on a fixed subset. All targets are model states, never CT GT."""
    small_field = geometric_subset(field, indices)
    pair = transport_subset(transport, small_field)
    q0, q1 = primary_state(small_field, 0.0), primary_state(small_field, 1.0)
    qu, qv = primary_state(small_field, u), primary_state(small_field, v)
    if not joint:
        q0, q1, qu, qv = map(detach_state, [q0, q1, qu, qv])
    move = lambda q, s, t: pair.transport(q, source_time=s, target_time=t)
    # Every iteration uses both directions; cycles are anchored at observed endpoints.
    a = move(q0, 0.0, u)
    b = move(q1, 1.0, u)
    alignment = 0.5 * (
        paired_alignment(pair, a, qu, joint) + paired_alignment(pair, b, qu, joint)
    )
    cycle = 0.5 * (
        distance(pair, move(a, u, 0.0), detach_state(q0))
        + distance(pair, move(b, u, 1.0), detach_state(q1))
    )
    # Compare a two-leg path with a detached direct path; alternate its anchor.
    source, s = (q0, 0.0) if u < 0.5 else (q1, 1.0)
    via = move(move(source, s, u), u, v)
    direct = move(source, s, v)
    path = distance(pair, via, detach_state(direct))
    direct_alignment = paired_alignment(pair, direct, qv, joint)
    alignment = 0.5 * (alignment + direct_alignment)
    return alignment, cycle, path


def make_train_step(options):
    """Build the step for the three new variants; reference methods use upstream code."""
    def train_step(canonical, field, optimizer, study, sample, smoothness_indices,
                   config, pairwise_cycle=None, *, iteration):
        assert float(sample["PhasePercent"]) == 50.0
        optimizer.zero_grad(set_to_none=True)
        device = str(field.xyz.device)
        slice_index = int(sample["SliceIndex"])
        target = tr.load_target_tensor(study, 50.0, slice_index,
                                      representation="raw", device=device)
        camera = tr.get_camera_for_slice(canonical, slice_index)
        view, endpoint = field.build_phase_state(0.5, use_checkpointing=True)
        rendered = canonical.runtime.render(
            camera, view, canonical.pipeline, canonical.background,
            train=True, iter=iteration,
        )["render"]
        image, metrics = tr.compute_reconstruction_loss(
            canonical, rendered, target,
            l1_weight=config.training.l1_weight, ssim_weight=config.training.ssim_weight,
        )
        magnitude = tr.compute_magnitude_loss(field, endpoint)
        current = field.build_subset_state(0.5, smoothness_indices)
        zero = field.build_subset_state(0.0, smoothness_indices)
        smoothness = tr.compute_temporal_smoothness_loss(field, current, zero)
        total = (image + config.training.magnitude_weight * magnitude
                 + config.training.smoothness_weight * smoothness)
        z = image.new_zeros(())
        correction, curvature, alignment, cycle, path, endpoint_image = [z] * 6
        rng = np.random.default_rng(config.training.seed + 900000 + iteration)
        u, v = map(float, rng.uniform(0.05, 0.95, size=2))
        subset_rng = np.random.default_rng(config.training.seed + 1002)
        positions = subset_rng.choice(len(smoothness_indices),
            size=min(len(smoothness_indices), options["geometry_gaussians"]), replace=False)
        indices = smoothness_indices.index_select(0, torch.as_tensor(
            np.sort(positions), device=smoothness_indices.device, dtype=torch.long))
        if isinstance(field, EndpointTrajectoryField):
            sf = geometric_subset(field, indices)
            probe = PairwiseTransport.__new__(PairwiseTransport)
            # geometry_vector needs only the field; no learned transport is instantiated.
            torch.nn.Module.__init__(probe)
            probe.field = sf
            qm = primary_state(sf, u)
            q1 = primary_state(sf, 1.0)
            displacement = probe.geometry_vector(qm)
            correction = (displacement - u * probe.geometry_vector(q1)).square().mean()
            h = options["curvature_step"]
            lo = probe.geometry_vector(primary_state(sf, u - h))
            hi = probe.geometry_vector(primary_state(sf, u + h))
            # Finite-difference acceleration in normalized geometry coordinates.
            curvature = ((hi - 2 * displacement + lo) / h**2).square().mean()
            total = total + options["correction_weight"] * correction
            total = total + options["curvature_weight"] * curvature
        warmup = options["endpoint_warmup"]
        pretrain_end = warmup + options["transport_pretrain"]
        joint = iteration > pretrain_end
        if pairwise_cycle is not None and iteration > warmup:
            alignment, cycle, path = consistency_losses(
                field, pairwise_cycle, indices, u, v, joint
            )
            middle = primary_state(field, u)
            if not joint:
                middle = detach_state(middle)
            destination = 0.0 if iteration % 2 == 0 else 1.0
            transported = pairwise_cycle.transport(
                middle, source_time=u, target_time=destination
            )
            endpoint_target = (target if destination == 1.0 else
                tr.load_target_tensor(study, 0.0, slice_index,
                                      representation="raw", device=device))
            auxiliary_render = canonical.runtime.render(
                camera, state_to_view(field, transported), canonical.pipeline,
                canonical.background, train=True, iter=iteration,
            )["render"]
            endpoint_image, _ = tr.compute_reconstruction_loss(
                canonical, auxiliary_render, endpoint_target,
                l1_weight=config.training.l1_weight,
                ssim_weight=config.training.ssim_weight,
            )
            # Pretraining is detached; joint coupling is ramped after it starts.
            scale = (1.0 if not joint else min(
                1.0, (iteration - pretrain_end) / options["joint_ramp"]
            ))
            auxiliary = (options["endpoint_image_weight"] * endpoint_image
                         + options["alignment_weight"] * alignment
                         + options["roundtrip_weight"] * cycle
                         + options["path_weight"] * path)
            total = total + scale * auxiliary
        else:
            scale = 0.0
        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite objective at iteration {iteration}")
        total.backward()
        parameters = list(field.model.parameters())
        if pairwise_cycle is not None:
            parameters += list(pairwise_cycle.parameters())
        grad = torch.nn.utils.clip_grad_norm_(parameters, config.training.max_gradient_norm)
        if not torch.isfinite(grad):
            raise FloatingPointError(f"Non-finite gradient at iteration {iteration}")
        optimizer.step()
        return {
            "TotalLoss": float(total.detach()), "ImageLoss": float(image.detach()),
            "L1": float(metrics["l1"].detach()), "SSIM": float(metrics["ssim"].detach()),
            "PSNR": float(metrics["psnr"].detach()), "GradientNorm": float(grad),
            "DeformationMagnitudeLoss": float(magnitude.detach()),
            "TemporalSmoothnessLoss": float(smoothness.detach()),
            "CorrectionLoss": float(correction.detach()),
            "CurvatureLoss": float(curvature.detach()),
            "AlignmentLoss": float(alignment.detach()),
            "RoundTripLoss": float(cycle.detach()), "PathLoss": float(path.detach()),
            "TransportEndpointImageLoss": float(endpoint_image.detach()),
            "AuxiliaryScale": scale, "JointTraining": int(joint), "LocalU": u, "LocalV": v,
        }
    return train_step


def field_class(method):
    return EndpointTrajectoryField if method in {"trajectory", "trajectory_cycle"} else DeformationField


def install_training_variant(method, options):
    """Select factories in this worker process only; source packages remain unchanged."""
    tr.DeformationField = field_class(method)
    if method in {"trajectory", "consistent_cycle", "trajectory_cycle"}:
        tr.train_step = make_train_step(options)
