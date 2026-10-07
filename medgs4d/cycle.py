from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch.utils.checkpoint import checkpoint

from .deformation import (
    DeformationField,
    PhaseDeformedGaussianView,
)


class PairwiseTransportMLP(torch.nn.Module):
    """Predict incremental Gaussian transport between two arbitrary times."""

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
                [
                    torch.nn.Linear(
                        current_dim,
                        hidden_dim,
                    ),
                    torch.nn.SiLU(),
                ]
            )
            current_dim = hidden_dim

        self.backbone = torch.nn.Sequential(
            *layers
        )

        self.output_layer = torch.nn.Linear(
            current_dim,
            3,
        )

        torch.nn.init.zeros_(
            self.output_layer.weight
        )
        torch.nn.init.zeros_(
            self.output_layer.bias
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.output_layer(
            self.backbone(inputs)
        )


@dataclass
class PairwiseCycleOutput:
    """Store Gaussian states and regularizers produced by one cycle."""

    reconstructed_canonical: dict[str, torch.Tensor]
    reconstructed_endpoint: dict[str, torch.Tensor]

    canonical_candidate_left: dict[str, torch.Tensor]
    canonical_candidate_middle: dict[str, torch.Tensor]

    endpoint_candidate_middle: dict[str, torch.Tensor]
    endpoint_candidate_right: dict[str, torch.Tensor]

    agreement_loss: torch.Tensor
    transport_loss: torch.Tensor


def state_to_view(
    field: DeformationField,
    state: Mapping[str, torch.Tensor],
) -> PhaseDeformedGaussianView:
    """Convert one Gaussian deformation state to a renderer-compatible view."""

    return PhaseDeformedGaussianView(
        field.canonical.gaussians,
        state["dynamic_xyz"],
        state["dynamic_m"],
    )


class PairwiseTransport(torch.nn.Module):
    """Transport a dynamic Gaussian state between arbitrary local times."""

    def __init__(
        self,
        field: DeformationField,
        *,
        hidden_dim: int = 128,
        hidden_layers: int = 3,
        time_frequencies: int = 2,
    ) -> None:
        super().__init__()

        self.field = field
        self.time_frequencies = int(
            time_frequencies
        )

        spatial_dim = int(
            field.spatial_features.shape[1]
        )

        time_dim = (
            1
            + 2 * self.time_frequencies
        )

        self.model = PairwiseTransportMLP(
            input_dim=(
                spatial_dim
                + 2 * time_dim
            ),
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        ).to(
            device=field.xyz.device,
            dtype=field.xyz.dtype,
        )

    @property
    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter
            in self.parameters()
        )

    def _encode_time(
        self,
        local_time: float,
    ) -> torch.Tensor:
        """Encode local cycle time without periodic wrapping."""

        value = torch.as_tensor(
            local_time,
            device=self.field.xyz.device,
            dtype=self.field.xyz.dtype,
        ).reshape(1)

        frequencies = (
            2.0
            ** torch.arange(
                self.time_frequencies,
                device=value.device,
                dtype=value.dtype,
            )
        )

        angles = (
            torch.pi
            * value.unsqueeze(-1)
            * frequencies
        )

        return torch.cat(
            [
                value,
                torch.sin(angles).flatten(),
                torch.cos(angles).flatten(),
            ],
            dim=0,
        )

    def _build_state(
        self,
        delta_xz: torch.Tensor,
        delta_m_logit: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build a complete dynamic state from canonical-relative residuals."""

        dynamic_xz = (
            self.field.xz
            + delta_xz
        )

        dynamic_m = torch.sigmoid(
            self.field.m_logits
            + delta_m_logit
        )

        dynamic_xyz = torch.stack(
            [
                dynamic_xz[:, 0],
                self.field.xyz[:, 1],
                dynamic_xz[:, 1],
            ],
            dim=-1,
        )

        relative = torch.cat(
            [
                delta_xz,
                delta_m_logit,
            ],
            dim=-1,
        )

        return {
            "relative_deformation": relative,
            "delta_xz": delta_xz,
            "delta_m_logit": delta_m_logit,
            "delta_m": (
                dynamic_m
                - self.field.m
            ),
            "dynamic_xyz": dynamic_xyz,
            "dynamic_m": dynamic_m,
        }

    def virtual_state(
        self,
        endpoint_state: Mapping[
            str,
            torch.Tensor,
        ],
        local_time: float,
    ) -> dict[str, torch.Tensor]:
        """Create an endpoint-derived virtual state by residual extrapolation."""

        delta_xz = (
            float(local_time)
            * endpoint_state[
                "delta_xz"
            ].detach()
        )

        delta_m_logit = (
            float(local_time)
            * endpoint_state[
                "delta_m_logit"
            ].detach()
        )

        return self._build_state(
            delta_xz,
            delta_m_logit,
        )

    def _spatial_features(
        self,
        source_state: Mapping[
            str,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        """Encode the geometry of the current source state."""

        source_xz = (
            self.field.xz
            + source_state["delta_xz"]
        )

        source_m = source_state[
            "dynamic_m"
        ]

        normalized = torch.cat(
            [
                (
                    source_xz
                    - self.field.xz_mean
                )
                / self.field.xz_scale,
                (
                    source_m
                    - self.field.m_mean
                )
                / self.field.m_scale,
            ],
            dim=-1,
        )

        return (
            self.field
            .encode_spatial_coordinates(
                normalized
            )
        )

    def _predict_increment(
        self,
        source_state: Mapping[
            str,
            torch.Tensor,
        ],
        *,
        source_time: float,
        target_time: float,
    ) -> torch.Tensor:
        """Predict one incremental transport residual."""

        spatial_features = (
            self._spatial_features(
                source_state
            )
        )

        source_encoding = (
            self._encode_time(
                source_time
            )
        )

        target_encoding = (
            self._encode_time(
                target_time
            )
        )

        chunks = []

        for start in range(
            0,
            spatial_features.shape[0],
            self.field.config.chunk_size,
        ):
            selected = spatial_features[
                start:
                start
                + self.field.config.chunk_size
            ]

            source_part = (
                source_encoding
                .unsqueeze(0)
                .expand(
                    selected.shape[0],
                    -1,
                )
            )

            target_part = (
                target_encoding
                .unsqueeze(0)
                .expand(
                    selected.shape[0],
                    -1,
                )
            )

            inputs = torch.cat(
                [
                    selected,
                    source_part,
                    target_part,
                ],
                dim=-1,
            )

            outputs = (
                checkpoint(
                    self.model,
                    inputs,
                    use_reentrant=False,
                )
                if torch.is_grad_enabled()
                else self.model(inputs)
            )

            chunks.append(outputs)

        raw_increment = torch.cat(
            chunks,
            dim=0,
        )

        delta_time = (
            float(target_time)
            - float(source_time)
        )

        return (
            delta_time
            * raw_increment
        )

    def transport(
        self,
        source_state: Mapping[
            str,
            torch.Tensor,
        ],
        *,
        source_time: float,
        target_time: float,
    ) -> dict[str, torch.Tensor]:
        """Transport one dynamic Gaussian state to another local time."""

        increment = (
            self._predict_increment(
                source_state,
                source_time=source_time,
                target_time=target_time,
            )
        )

        delta_xz = (
            source_state["delta_xz"]
            + increment[:, :2]
        )

        delta_m_logit = (
            source_state[
                "delta_m_logit"
            ]
            + increment[:, 2:3]
        )

        return self._build_state(
            delta_xz,
            delta_m_logit,
        )

    def geometry_vector(
        self,
        state: Mapping[
            str,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        """Express Gaussian geometry in normalized displacement units."""

        return torch.cat(
            [
                state["delta_xz"]
                / self.field.xz_scale,
                state["delta_m"]
                / self.field.m_scale,
            ],
            dim=-1,
        )

    def geometry_distance(
        self,
        first: Mapping[
            str,
            torch.Tensor,
        ],
        second: Mapping[
            str,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        """Measure normalized geometric disagreement between two states."""

        first_vector = (
            self.geometry_vector(first)
        )

        second_vector = (
            self.geometry_vector(second)
        )

        return (
            first_vector
            - second_vector
        ).square().mean()

    def blend(
        self,
        first: Mapping[
            str,
            torch.Tensor,
        ],
        second: Mapping[
            str,
            torch.Tensor,
        ],
        *,
        first_source_time: float,
        second_source_time: float,
        endpoint_time: float,
    ) -> dict[str, torch.Tensor]:
        """Blend two endpoint candidates using inverse temporal distance."""

        first_distance = max(
            abs(
                float(first_source_time)
                - float(endpoint_time)
            ),
            1e-3,
        )

        second_distance = max(
            abs(
                float(second_source_time)
                - float(endpoint_time)
            ),
            1e-3,
        )

        first_inverse = (
            1.0 / first_distance
        )

        second_inverse = (
            1.0 / second_distance
        )

        denominator = (
            first_inverse
            + second_inverse
        )

        first_weight = (
            first_inverse
            / denominator
        )

        second_weight = (
            second_inverse
            / denominator
        )

        delta_xz = (
            first_weight
            * first["delta_xz"]
            + second_weight
            * second["delta_xz"]
        )

        delta_m_logit = (
            first_weight
            * first["delta_m_logit"]
            + second_weight
            * second["delta_m_logit"]
        )

        return self._build_state(
            delta_xz,
            delta_m_logit,
        )


def build_pairwise_cycle(
    transport: PairwiseTransport,
    *,
    endpoint_state: Mapping[
        str,
        torch.Tensor,
    ],
    middle_state: Mapping[
        str,
        torch.Tensor,
    ],
    u1: float,
    u2: float,
    u3: float,
) -> PairwiseCycleOutput:
    """Build one three-time pairwise Gaussian cycle."""

    left_state = (
        transport.virtual_state(
            endpoint_state,
            u1,
        )
    )

    right_state = (
        transport.virtual_state(
            endpoint_state,
            u3,
        )
    )

    canonical_candidate_left = (
        transport.transport(
            left_state,
            source_time=u1,
            target_time=0.0,
        )
    )

    canonical_candidate_middle = (
        transport.transport(
            middle_state,
            source_time=u2,
            target_time=0.0,
        )
    )

    endpoint_candidate_middle = (
        transport.transport(
            middle_state,
            source_time=u2,
            target_time=1.0,
        )
    )

    endpoint_candidate_right = (
        transport.transport(
            right_state,
            source_time=u3,
            target_time=1.0,
        )
    )

    reconstructed_canonical = (
        transport.blend(
            canonical_candidate_left,
            canonical_candidate_middle,
            first_source_time=u1,
            second_source_time=u2,
            endpoint_time=0.0,
        )
    )

    reconstructed_endpoint = (
        transport.blend(
            endpoint_candidate_middle,
            endpoint_candidate_right,
            first_source_time=u2,
            second_source_time=u3,
            endpoint_time=1.0,
        )
    )

    agreement_loss = 0.5 * (
        transport.geometry_distance(
            canonical_candidate_left,
            canonical_candidate_middle,
        )
        + transport.geometry_distance(
            endpoint_candidate_middle,
            endpoint_candidate_right,
        )
    )

    transport_loss = 0.25 * (
        transport.geometry_distance(
            left_state,
            canonical_candidate_left,
        )
        + transport.geometry_distance(
            middle_state,
            canonical_candidate_middle,
        )
        + transport.geometry_distance(
            middle_state,
            endpoint_candidate_middle,
        )
        + transport.geometry_distance(
            right_state,
            endpoint_candidate_right,
        )
    )

    return PairwiseCycleOutput(
        reconstructed_canonical=(
            reconstructed_canonical
        ),
        reconstructed_endpoint=(
            reconstructed_endpoint
        ),
        canonical_candidate_left=(
            canonical_candidate_left
        ),
        canonical_candidate_middle=(
            canonical_candidate_middle
        ),
        endpoint_candidate_middle=(
            endpoint_candidate_middle
        ),
        endpoint_candidate_right=(
            endpoint_candidate_right
        ),
        agreement_loss=agreement_loss,
        transport_loss=transport_loss,
    )
