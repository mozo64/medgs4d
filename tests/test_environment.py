from pathlib import Path
from types import SimpleNamespace
import pytest
from benchmark.environment import validate_environment


def test_gpu_selection_respects_kernel_visibility(tmp_path, monkeypatch):
    (tmp_path / 'benchmark').mkdir()
    (tmp_path / 'benchmark/worker.py').touch()
    module = SimpleNamespace(__file__=str(tmp_path / 'benchmark/metrics.py'))
    monkeypatch.setattr('torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('torch.cuda.device_count', lambda: 2)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,5')
    assert validate_environment(tmp_path, 1, module) == '5'
    with pytest.raises(ValueError, match='GPU_INDEX'):
        validate_environment(tmp_path, -1, module)
    module.__file__ = str(tmp_path / 'other/metrics.py')
    with pytest.raises(RuntimeError, match='Kernel imported'):
        validate_environment(tmp_path, 0, module)


def test_temporal_data_imports_do_not_load_cuda_models():
    import subprocess, sys
    root = Path(__file__).resolve().parents[1]
    code = '''import sys
import scene.volume_data
import models.trajectory
assert 'scene.gaussian_model' not in sys.modules
assert 'diff_gaussian_rasterization' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], cwd=root, check=True)
