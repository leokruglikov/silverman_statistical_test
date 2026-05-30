# Depict Silverman

Small package and marimo notebooks for exploring Silverman's multimodality test.

## Structure

```text
silverman/
  __init__.py
  silverman_test.py
  utils.py
notebooks/
  silverman_edge_cases.mo.py
  silverman_edge_cases_parallel.mo.py
  ...
silverman_test.py
silverman_utils.py
requirements.txt
pyproject.toml
```

Notes:

- `silverman/` is the package-first code.
- `notebooks/` contains the marimo explorations.
- top-level `silverman_test.py` and `silverman_utils.py` are compatibility shims.

## Install

You can use the project either as a local repo or as a small package.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or install it in editable mode:

```bash
pip install -e .
```

## Main API

The main entry point is:

```python
from silverman import silverman_test
```

or:

```python
from silverman.silverman_test import silverman_test
```

### Example

```python
import numpy as np
from silverman import silverman_test

rng = np.random.default_rng(0)
x = np.concatenate([
    rng.normal(-2.0, 0.7, 150),
    rng.normal(2.0, 0.7, 150),
])

result = silverman_test(
    x,
    k_modes=1,
    method="silverman",
    n_boot=100,
    parallel=False,
)

print(result.pvalue)
print(result.h_crit)
print(result.summary())
```

## API Details

`silverman_test(...)` supports:

- `method="silverman"`: plain Silverman bootstrap
- `method="variance_corrected"`: variance-corrected version
- `method="hall_york"`: Hall-York adjusted version
- `parallel=True`: parallel bootstrap with `joblib`
- `parallel=False`: sequential bootstrap

Important arguments:

- `data`: one-dimensional sample
- `k_modes`: null hypothesis is at most `k_modes` modes
- `n_boot`: number of bootstrap draws
- `alpha`: Hall-York correction level
- `n_grid`, `margin`, `max_iter`, `tol`, `h_low_factor`: search/KDE controls

## Result Object

The returned object is `SilvermanTestResult`. It is intentionally lightweight and statsmodels-like:

- `result.pvalue`
- `result.h_crit`
- `result.h_stars`
- `result.search_history`
- `result.x_grid`
- `result.method`
- `result.summary()`

For Hall-York runs it also contains:

- `result.alpha`
- `result.lambda_alpha`
- `result.pvalue_raw`

## Notebooks

Run the notebooks from the repo root:

```bash
marimo edit notebooks/silverman_edge_cases.mo.py
```

or the parallel version:

```bash
marimo edit notebooks/silverman_edge_cases_parallel.mo.py
```

## Scope

This is still a small experimental project, not a full statistical library. The goal is to keep the implementation simple, readable, and close to the notebook logic that motivated it.
