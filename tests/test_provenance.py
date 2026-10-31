from pathlib import Path
import subprocess
import pytest
from benchmark.provenance import record_sources,verify_sources


def test_source_changes_and_added_files_are_rejected(tmp_path):
    subprocess.run(['git','init',str(tmp_path)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(tmp_path),'-c','user.name=Test','-c','user.email=test@example.com','commit','--allow-empty','-m','test'],check=True,capture_output=True)
    (tmp_path/'benchmark').mkdir()
    code=tmp_path/'benchmark/worker.py';code.write_text('x=1\n')
    manifest=tmp_path/'output/manifest.json'
    record_sources(tmp_path,manifest);verify_sources(tmp_path,manifest)
    code.write_text('x=2\n')
    with pytest.raises(RuntimeError,match='changed'):verify_sources(tmp_path,manifest)
    with pytest.raises(RuntimeError,match='changed'):record_sources(tmp_path,manifest)
    code.write_text('x=1\n')
    (tmp_path/'benchmark/new.py').write_text('x=3\n')
    with pytest.raises(RuntimeError,match='changed'):verify_sources(tmp_path,manifest)
