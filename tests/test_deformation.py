from types import SimpleNamespace

import torch

from medgs4d.config import DeformationConfig
from medgs4d.deformation import DeformationField


def fake_canonical(count: int = 128):
    xyz = torch.randn(count, 3)
    m_logits = torch.randn(count, 1)
    return SimpleNamespace(
        xyz=xyz,
        xz=xyz[:, [0, 2]],
        m_logits=m_logits,
        m=torch.sigmoid(m_logits),
        gaussians=SimpleNamespace(),
    )


def test_v2_parameter_count_matches_notebook() -> None:
    field = DeformationField(
        fake_canonical(),
        DeformationConfig(
            spatial_frequencies=4,
            phase_frequencies=2,
            hidden_dim=256,
            hidden_layers=4,
            chunk_size=32,
        ),
        canonical_phase=20,
    )
    assert field.parameter_count == 206_339


def test_canonical_phase_has_zero_deformation() -> None:
    field = DeformationField(
        fake_canonical(), DeformationConfig(chunk_size=32), canonical_phase=20
    )
    relative = field.predict_relative_deformation(0.2, use_checkpointing=False)
    assert torch.count_nonzero(relative).item() == 0


def test_deformation_output_shapes() -> None:
    canonical = fake_canonical(35)
    field = DeformationField(
        canonical, DeformationConfig(chunk_size=8), canonical_phase=20
    )
    view, state = field.build_phase_state(0.5, use_checkpointing=False)
    assert view.get_xyz.shape == (35, 3)
    assert view.get_m.shape == (35, 1)
    assert state["delta_xz"].shape == (35, 2)
    assert state["delta_m_logit"].shape == (35, 1)


def test_chunked_and_single_chunk_forward_match() -> None:
    canonical = fake_canonical(41)
    small = DeformationField(
        canonical, DeformationConfig(chunk_size=7), canonical_phase=20, seed=9
    )
    large = DeformationField(
        canonical, DeformationConfig(chunk_size=1000), canonical_phase=20, seed=9
    )
    large.model.load_state_dict(small.model.state_dict())
    with torch.no_grad():
        a = small.predict_relative_deformation(0.6, use_checkpointing=False)
        b = large.predict_relative_deformation(0.6, use_checkpointing=False)
    assert torch.allclose(a, b, atol=1e-7)
