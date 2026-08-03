import json
from pathlib import Path


def test_results_browser_is_valid_and_code_cells_compile() -> None:
    """Keep the thin results notebook valid and free of syntax errors."""

    path = Path(__file__).resolve().parents[1] / "notebooks" / "results_browser.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec")
