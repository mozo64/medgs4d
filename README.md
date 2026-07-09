# MedGS4D

Working repository for 4D medical Gaussian Splatting experiments on dynamic CT data.

The project is based on [GMUM MedGS](https://github.com/gmum/MedGS) and focuses on the practical preparation layer for 4D experiments: CUDA environment setup, TCIA 4D-Lung download, DICOM inspection, respiratory-phase exploration, and preliminary visualization.

## Repository structure

```text
medgs4d/
├── README.md
├── docs/
│   └── medgs_clean_install_guide.md
├── notebooks/
│   └── explore_4d_lung.ipynb
└── scripts/
    ├── download_all_series.py
    └── pycharm_env_check.py
```

## Files

| Path | Purpose |
|---|---|
| `docs/medgs_clean_install_guide.md` | Environment setup notes for the Ubuntu/CUDA server used to run and test MedGS. |
| `notebooks/explore_4d_lung.ipynb` | Exploratory notebook for TCIA 4D-Lung metadata, DICOM unpacking, respiratory phases, and slice visualization. |
| `scripts/download_all_series.py` | Downloader for TCIA 4D-Lung series through the NBIA API. It saves one ZIP per DICOM series and logs download status. |
| `scripts/pycharm_env_check.py` | Smoke test for the MedGS Python/CUDA environment. |

## Upstream dependency

The upstream MedGS repository should be cloned separately:

```text
https://github.com/gmum/MedGS
```

This repository contains the auxiliary workflow around MedGS, not a vendored copy of the upstream code.

## Data source

The initial dynamic CT dataset is TCIA 4D-Lung:

```text
https://www.cancerimagingarchive.net/collection/4d-lung/
```

The downloaded data should be stored outside Git. The scripts and notebook currently define local path constants near the top of the file; edit these paths to match your own server or workstation layout.

Examples of constants to adjust:

```python
ROOT = Path(".../data/tcia_4d_lung")
MEDGS_ROOT = Path("...")
```

Expected data layout:

```text
tcia_4d_lung/
├── metadata/
│   ├── patients.json
│   ├── series.json
│   ├── series_summary.csv
│   └── download_log.csv
├── raw/
│   ├── series_zips/
│   └── dicom_by_series/
└── processed/
```

## Notebook environment

The notebook uses the same Python/CUDA environment as MedGS. After creating and validating the MedGS environment, install the notebook-related packages into that environment:

```bash
pip install jupyter ipykernel pydicom pandas ipywidgets
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install \
  --prefix "$CONDA_PREFIX" \
  --name medgs38 \
  --display-name "Python (medgs38 MedGS)"
```

In PyCharm or Jupyter, select:

```text
Python (medgs38 MedGS)
```

or directly use the Python interpreter from the MedGS environment.

## Running the TCIA downloader

Before running the downloader, create or obtain a TCIA series manifest:

```text
data/tcia_4d_lung/metadata/series.json
```

Then edit the `ROOT` constant in `scripts/download_all_series.py` so that it points to your local TCIA 4D-Lung data directory.

Run:

```bash
python scripts/download_all_series.py
```

The script writes downloaded ZIP files under:

```text
raw/series_zips/
```

and logs progress to:

```text
metadata/download_log.csv
```

For long downloads, run the command inside `tmux`:

```bash
tmux new -s tcia4dlung
python scripts/download_all_series.py
```

Reconnect after SSH interruption with:

```bash
tmux attach -t tcia4dlung
```
