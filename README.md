# Depict Silverman

Small notebooks and utilities for exploring Silverman's test for multimodality.

The project is intentionally lightweight. It contains:

- `silverman_utils.py`: small helpers for simulation, KDE mode counting, critical-bandwidth search, and Silverman-style bootstrap variants
- `silverman_edge_cases.mo.py`: a simple marimo notebook for illustrating where the test works well and where it becomes fragile
- `silverman_edge_cases_parallel.mo.py`: the same notebook with some heavier simulation cells parallelized using `joblib`
- other `*.mo.py` notebooks used for earlier experiments and illustrations

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Launch a notebook with marimo:

```bash
marimo edit silverman_edge_cases.mo.py
```

or the parallel version:

```bash
marimo edit silverman_edge_cases_parallel.mo.py
```

## Notes

- The repository is focused on experimentation and illustration rather than packaging.
- Generated HTML exports and local virtualenv files are ignored by Git.
