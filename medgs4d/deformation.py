from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from .canonical import CanonicalAssets
from .config import DeformationConfig


class DeformationMLP(torch.nn.Module):
    """Predict x, z, and m-logit residuals from spatial and phase features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
    ) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.extend(
                [torch.nn.Linear(current_dim, hidden_dim), torch.nn.SiLU()]
            )
            current_dim = hidden_dim
        self.backbone = torch.nn.Sequential(*layers)
        self.output_layer = torch.nn.Linear(current_dim, 3)
        torch.nn.init.zeros_(self.output_layer.weight)
        torch.nn.init.zeros_(self.output_layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.backbone(inputs))


class PhaseDeformedGaussianView:
    """Expose dynamic xyz and m while forwarding fixed canonical properties."""

    def __init__(
        self,
        canonical_model: Any,
        dynamic_xyz: torch.Tensor,
        dynamic_m: torch.Tensor,
    ) -> None:
        self._canonical_model = canonical_model
        self._dynamic_xyz = dynamic_xyz
        self._dynamic_m = dynamic_m

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._dynamic_xyz

    @property
    def get_m(self) -> torch.Tensor:
        return self._dynamic_m

    def __getattr__(self, name: str) -> Any:
        return getattr(self._canonical_model, name)


@dataclass
class DeformationState:
    """Store dynamic Gaussian geometry and its residual components."""

    relative_deformation: torch.Tensor
    delta_xz: torch.Tensor
    delta_m_logit: torch.Tensor
    delta_m: torch.Tensor
    dynamic_xyz: torch.Tensor
    dynamic_m: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "relative_deformation": self.relative_deformation,
            "delta_xz": self.delta_xz,
            "delta_m_logit": self.delta_m_logit,
            "delta_m": self.delta_m,
            "dynamic_xyz": self.dynamic_xyz,
            "dynamic_m": self.dynamic_m,
        }


class DeformationField:
    """Predict exactly anchored phase-dependent x, z, and m deformations."""

    def __init__(
        self,
        canonical: CanonicalAssets,
        config: DeformationConfig,
        canonical_phase: float,
        *,
        seed: int = 42,
    ) -> None:
        self.canonical = canonical
        self.config = config
        self.canonical_phase = float(canonical_phase)
        self.xyz = canonical.xyz
        self.xz = canonical.xz
        self.m_logits = canonical.m_logits
        self.m = canonical.m
        self.xz_mean = self.xz.mean(dim=0, keepdim=True)
        self.xz_scale = self.xz.std(
            dim=0, keepdim=True, unbiased=False
        ).clamp_min(1e-6)
        self.m_mean = self.m.mean(dim=0, keepdim=True)
        self.m_scale = self.m.std(
            dim=0, keepdim=True, unbiased=False
        ).clamp_min(1e-6)
        normalized = torch.cat(
            [
                (self.xz - self.xz_mean) / self.xz_scale,
                (self.m - self.m_mean) / self.m_scale,
            ],
            dim=-1,
        ).detach()
        self.normalized_coordinates = normalized
        self.spatial_features = self.encode_spatial_coordinates(normalized).detach()
        phase_dim = int(self.encode_respiratory_phase(0.0).numel())
        input_dim = int(self.spatial_features.shape[1]) + phase_dim

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.model = DeformationMLP(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            hidden_layers=config.hidden_layers,
        ).to(device=self.xyz.device, dtype=self.xyz.dtype)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def encode_spatial_coordinates(
        self,
        normalized_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Encode normalized x, z, and m coordinates with Fourier features."""

        if self.config.spatial_frequencies == 0:
            return normalized_coordinates
        frequencies = 2.0 ** torch.arange(
            self.config.spatial_frequencies,
            device=normalized_coordinates.device,
            dtype=normalized_coordinates.dtype,
        )
        angles = (
            torch.pi
            * normalized_coordinates.unsqueeze(-1)
            * frequencies.view(1, 1, -1)
        )
        return torch.cat(
            [
                normalized_coordinates,
                torch.sin(angles).flatten(start_dim=1),
                torch.cos(angles).flatten(start_dim=1),
            ],
            dim=-1,
        )

    def encode_respiratory_phase(self, respiratory_time: float) -> torch.Tensor:
        """Encode normalized respiratory phase with cyclic Fourier features."""

        value = torch.as_tensor(
            respiratory_time,
            device=self.xyz.device,
            dtype=self.xyz.dtype,
        )
        frequencies = 2.0 ** torch.arange(
            self.config.phase_frequencies,
            device=self.xyz.device,
            dtype=self.xyz.dtype,
        )
        angles = 2.0 * torch.pi * torch.remainder(value, 1.0) * frequencies
        return torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1).flatten()

    def build_mlp_inputs(
        self,
        spatial_features: torch.Tensor,
        respiratory_time: float,
    ) -> torch.Tensor:
        """Combine per-Gaussian spatial features with one phase encoding."""

        phase = self.encode_respiratory_phase(respiratory_time)
        return torch.cat(
            [spatial_features, phase.unsqueeze(0).expand(spatial_features.shape[0], -1)],
            dim=-1,
        )

    def predict_relative_deformation(
        self,
        respiratory_time: float,
        *,
        spatial_features: torch.Tensor | None = None,
        use_checkpointing: bool = True,
    ) -> torch.Tensor:
        """Predict deformation relative to the selected canonical phase."""

        features = self.spatial_features if spatial_features is None else spatial_features
        canonical_time = self.canonical_phase / 100.0
        chunks = []
        for start in range(0, features.shape[0], self.config.chunk_size):
            selected = features[start : start + self.config.chunk_size]
            phase_inputs = self.build_mlp_inputs(selected, respiratory_time)
            canonical_inputs = self.build_mlp_inputs(selected, canonical_time)
            joint_inputs = torch.cat([phase_inputs, canonical_inputs], dim=0)
            outputs = (
                checkpoint(self.model, joint_inputs, use_reentrant=False)
                if use_checkpointing and torch.is_grad_enabled()
                else self.model(joint_inputs)
            )
            count = selected.shape[0]
            chunks.append(outputs[:count] - outputs[count:])
        return torch.cat(chunks, dim=0)

    def build_phase_state(
        self,
        respiratory_time: float,
        *,
        use_checkpointing: bool = True,
    ) -> tuple[PhaseDeformedGaussianView, dict[str, torch.Tensor]]:
        """Build renderer-compatible Gaussian geometry for one respiratory phase."""

        relative = self.predict_relative_deformation(
            respiratory_time, use_checkpointing=use_checkpointing
        )
        delta_xz = relative[:, :2]
        delta_m_logit = relative[:, 2:3]
        dynamic_xz = self.xz + delta_xz
        dynamic_m = torch.sigmoid(self.m_logits + delta_m_logit)
        dynamic_xyz = torch.stack(
            [dynamic_xz[:, 0], self.xyz[:, 1], dynamic_xz[:, 1]], dim=-1
        )
        state = DeformationState(
            relative_deformation=relative,
            delta_xz=delta_xz,
            delta_m_logit=delta_m_logit,
            delta_m=dynamic_m - self.m,
            dynamic_xyz=dynamic_xyz,
            dynamic_m=dynamic_m,
        )
        return (
            PhaseDeformedGaussianView(
                self.canonical.gaussians, dynamic_xyz, dynamic_m
            ),
            state.as_dict(),
        )

    def build_subset_state(
        self,
        respiratory_time: float,
        gaussian_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build a deformation state for a fixed Gaussian subset."""

        features = self.spatial_features.index_select(0, gaussian_indices)
        relative = self.predict_relative_deformation(
            respiratory_time,
            spatial_features=features,
            use_checkpointing=False,
        )
        delta_xz = relative[:, :2]
        delta_m_logit = relative[:, 2:3]
        base_m_logits = self.m_logits.index_select(0, gaussian_indices)
        base_m = self.m.index_select(0, gaussian_indices)
        dynamic_m = torch.sigmoid(base_m_logits + delta_m_logit)
        return {
            "relative_deformation": relative,
            "delta_xz": delta_xz,
            "delta_m_logit": delta_m_logit,
            "delta_m": dynamic_m - base_m,
            "dynamic_m": dynamic_m,
        }

    def normalization_dict(self) -> dict[str, object]:
        """Return saved normalization statistics for reproducibility."""

        return {
            "canonical_phase": self.canonical_phase,
            "gaussian_count": int(self.xyz.shape[0]),
            "xz_mean": self.xz_mean.squeeze(0).detach().cpu().tolist(),
            "xz_scale": self.xz_scale.squeeze(0).detach().cpu().tolist(),
            "m_mean": float(self.m_mean.item()),
            "m_scale": float(self.m_scale.item()),
            "spatial_frequencies": self.config.spatial_frequencies,
            "phase_frequencies": self.config.phase_frequencies,
            "input_dimension": int(self.model.backbone[0].in_features),
            "parameter_count": self.parameter_count,
        }
