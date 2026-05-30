"""Polished Silverman-test entry point."""

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed

from . import utils


@dataclass
class SilvermanTestResult:
    """Lightweight result object in a statsmodels-like style."""

    method: str
    k_modes: int
    n_samples: int
    n_boot: int
    h_crit: float
    pvalue: float
    h_stars: np.ndarray
    search_history: list[float]
    x_grid: np.ndarray
    alpha: float | None = None
    lambda_alpha: float | None = None
    pvalue_raw: float | None = None
    parallel: bool = False
    n_jobs: int | None = None

    @property
    def statistic(self):
        return self.h_crit

    def summary(self):
        lines = [
            "SilvermanTestResult",
            f"method={self.method}",
            f"k_modes={self.k_modes}",
            f"n_samples={self.n_samples}",
            f"n_boot={self.n_boot}",
            f"h_crit={self.h_crit:.6f}",
            f"pvalue={self.pvalue:.6f}",
            f"parallel={self.parallel}",
        ]
        if self.method == "hall_york":
            lines.append(f"alpha={self.alpha:.4f}")
            lines.append(f"lambda_alpha={self.lambda_alpha:.6f}")
            lines.append(f"pvalue_raw={self.pvalue_raw:.6f}")
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()


def _resolve_variant(method):
    if method == "silverman":
        return utils.plain_silverman_variant
    if method == "variance_corrected":
        return utils.corrected_var_silverman_variant
    if method == "hall_york":
        return utils.hall_york_silverman_variant
    raise ValueError(
        "method must be one of {'silverman', 'variance_corrected', 'hall_york'}"
    )


def _run_one_boot(seed, data, k_modes, h_crit, method, alpha, search_kwargs):
    rng = np.random.default_rng(seed)
    if method == "silverman":
        xb = utils.bootstrap_resample(data, rng=rng) + rng.normal(
            0.0, h_crit, size=len(data)
        )
    else:
        xb = utils._variance_corrected_bootstrap_sample(data, h_crit, rng)

    out_b = utils.search_h_critical(xb, k_modes, **search_kwargs)
    return out_b["h_crit"]


def silverman_test(
    data,
    k_modes=1,
    method="silverman",
    n_boot=80,
    alpha=0.05,
    parallel=False,
    n_jobs=-1,
    random_state=None,
    n_grid=1000,
    margin=0.0,
    max_iter=60,
    tol=1e-8,
    h_low_factor=1e-6,
):
    """Run a Silverman-style multimodality test.

    Parameters
    ----------
    data : array-like
        One-dimensional sample.
    k_modes : int
        Null hypothesis is that the density has at most ``k_modes`` modes.
    method : {"silverman", "variance_corrected", "hall_york"}
        Variant of the Silverman-style bootstrap.
    n_boot : int
        Number of bootstrap draws.
    alpha : float
        Hall-York level. Only used when ``method="hall_york"``.
    parallel : bool
        If True, parallelize bootstrap replications with joblib.
    n_jobs : int
        Number of workers when ``parallel=True``.
    random_state : int or numpy.random.Generator or None
        Random seed or generator.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError("data must be one-dimensional")
    if len(data) < 2:
        raise ValueError("data must contain at least two observations")

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    x_grid = utils.make_grid(data, n_grid=n_grid, margin=margin)
    search_kwargs = {
        "x_grid": x_grid,
        "max_iter": max_iter,
        "tol": tol,
        "h_low_factor": h_low_factor,
    }

    variant_fn = _resolve_variant(method)
    base = utils.search_h_critical(data, k_modes, **search_kwargs)
    h_crit = base["h_crit"]

    seeds = rng.integers(0, np.iinfo(np.uint32).max, size=n_boot, dtype=np.uint32)
    if parallel:
        h_stars = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_one_boot)(
                int(seed),
                data,
                k_modes,
                h_crit,
                method,
                alpha,
                search_kwargs,
            )
            for seed in seeds
        )
    else:
        h_stars = [
            _run_one_boot(
                int(seed),
                data,
                k_modes,
                h_crit,
                method,
                alpha,
                search_kwargs,
            )
            for seed in seeds
        ]

    h_stars = np.asarray(h_stars, dtype=float)
    pvalue_raw = float(np.mean(h_stars >= h_crit))
    if method == "hall_york":
        lambda_alpha = utils.hall_york_lambda(alpha)
        pvalue = float(np.mean(h_stars >= lambda_alpha * h_crit))
    else:
        lambda_alpha = None
        pvalue = pvalue_raw

    return SilvermanTestResult(
        method=method,
        k_modes=int(k_modes),
        n_samples=len(data),
        n_boot=int(n_boot),
        h_crit=float(h_crit),
        pvalue=float(pvalue),
        h_stars=h_stars,
        search_history=base["history"],
        x_grid=x_grid,
        alpha=float(alpha) if method == "hall_york" else None,
        lambda_alpha=lambda_alpha,
        pvalue_raw=pvalue_raw if method == "hall_york" else None,
        parallel=bool(parallel),
        n_jobs=n_jobs if parallel else None,
    )
