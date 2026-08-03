from pathlib import Path
import os

import pytest


@pytest.mark.worf
@pytest.mark.skipif(
    "MEDGS4D_TEST_RUN_DIR" not in os.environ,
    reason="Set MEDGS4D_TEST_RUN_DIR to enable WORF integration validation.",
)
def test_existing_run_loads_models_and_renders() -> None:
    import torch

    from medgs4d.results import load_run, load_run_models
    from medgs4d.visualization import render_run_slice

    run = load_run(run_dir=Path(os.environ["MEDGS4D_TEST_RUN_DIR"]))
    canonical, field = load_run_models(run, device="cuda")
    assert torch.cuda.is_available()
    images = render_run_slice(
        run,
        canonical,
        field,
        phase=float(run.study.phases[0]),
        slice_index=run.study.slice_count // 2,
    )
    assert images["ground_truth"].shape == images["dynamic"].shape
