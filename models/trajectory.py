"""Endpoint-anchored trajectory model for phases 0–50%."""
from __future__ import annotations
import torch
from torch.utils.checkpoint import checkpoint
from models.deformation import DeformationField


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
        if float(canonical_phase) != 0.0:
            raise ValueError("Endpoint trajectory requires canonical phase 0%")
        torch.manual_seed(seed)
        self.model = EndpointTrajectoryMLP(
            self.spatial_features.shape[1], config.hidden_dim, config.hidden_layers
        ).to(device=self.xyz.device, dtype=self.xyz.dtype)

    def predict_relative_deformation(self, respiratory_time, *,
                                    spatial_features=None, use_checkpointing=True):
        u = float(respiratory_time) / 0.5
        if not -1e-7 <= u <= 1.0 + 1e-7:
            raise ValueError("Trajectory supports phases 0–50%")
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
