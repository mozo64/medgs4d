import json
from pathlib import Path


def test_sparse_joint_direct_notebook_is_valid_and_code_cells_compile() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "05_sparse_joint_medgs_direct.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec")
