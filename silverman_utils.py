"""Simple utilities for Silverman-test experiments.

These helpers are intentionally minimal and easy to modify.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from tqdm.auto import tqdm


def plain_silverman_variant(data, k_modes, b_boot=80, rng=None, **search_kwargs):
    """Minimal Silverman-style bootstrap test for ``k_modes`` modes."""
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=float)
    base = search_h_critical(data, k_modes, **search_kwargs)
    h_crit = base["h_crit"]

    h_stars = []
    for _ in range(b_boot):
        xb = bootstrap_resample(data, rng=rng) + rng.normal(0.0, h_crit, size=len(data))
        out_b = search_h_critical(xb, k_modes, **search_kwargs)
        h_stars.append(out_b["h_crit"])

    h_stars = np.asarray(h_stars)
    return {
        "h_crit": float(h_crit),
        "p_value": float(np.mean(h_stars >= h_crit)),
        "h_stars": h_stars,
        "search_history": base["history"],
        "x_grid": base["x_grid"],
    }


def _variance_corrected_bootstrap_sample(data, h_crit, rng):
    """Bootstrap sample with Silverman's variance correction."""
    data = np.asarray(data, dtype=float)
    x_mean = np.mean(data)
    x_std = np.std(data, ddof=1)

    if x_std <= 0:
        raise ValueError("Cannot use variance correction when data variance is zero.")

    base = bootstrap_resample(data, rng=rng)
    noise = rng.normal(0.0, h_crit, size=len(data))
    correction = np.sqrt(1.0 + h_crit**2 / x_std**2)
    return x_mean + ((base - x_mean) + noise) / correction


def corrected_var_silverman_variant(data, k_modes, b_boot=80, rng=None, **search_kwargs):
    """Silverman-style bootstrap with variance correction."""
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=float)
    base = search_h_critical(data, k_modes, **search_kwargs)
    h_crit = base["h_crit"]

    h_stars = []
    for _ in range(b_boot):
        xb = _variance_corrected_bootstrap_sample(data, h_crit, rng)
        out_b = search_h_critical(xb, k_modes, **search_kwargs)
        h_stars.append(out_b["h_crit"])

    h_stars = np.asarray(h_stars)
    return {
        "h_crit": float(h_crit),
        "p_value": float(np.mean(h_stars >= h_crit)),
        "h_stars": h_stars,
        "search_history": base["history"],
        "x_grid": base["x_grid"],
    }


def hall_york_lambda(alpha):
    """Hall-York correction factor as used in the notebook."""
    alpha = float(alpha)
    num = (
        0.94029 * alpha**3
        - 1.59914 * alpha**2
        + 0.17695 * alpha
        + 0.48971
    )
    den = (
        alpha**3
        - 1.77793 * alpha**2
        + 0.36162 * alpha
        + 0.42423
    )
    return float(num / den)


def hall_york_silverman_variant(
    data,
    k_modes,
    b_boot=80,
    alpha=0.05,
    rng=None,
    **search_kwargs,
):
    """Variance-corrected Silverman bootstrap with Hall-York adjustment."""
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=float)
    base = search_h_critical(data, k_modes, **search_kwargs)
    h_crit = base["h_crit"]

    h_stars = []
    for _ in range(b_boot):
        xb = _variance_corrected_bootstrap_sample(data, h_crit, rng)
        out_b = search_h_critical(xb, k_modes, **search_kwargs)
        h_stars.append(out_b["h_crit"])

    h_stars = np.asarray(h_stars)
    lambda_alpha = hall_york_lambda(alpha)
    p_value_raw = float(np.mean(h_stars >= h_crit))
    p_value = float(np.mean(h_stars >= lambda_alpha * h_crit))

    return {
        "h_crit": float(h_crit),
        "p_value": p_value,
        "p_value_raw": p_value_raw,
        "lambda_alpha": lambda_alpha,
        "alpha": float(alpha),
        "h_stars": h_stars,
        "search_history": base["history"],
        "x_grid": base["x_grid"],
    }


def gen_mixture(n_samples, means, sigmas=None, weights=None, rng=None, return_components=False):
    """Generate a 1D Gaussian mixture sample."""
    if rng is None:
        rng = np.random.default_rng()

    means = np.asarray(means, dtype=float)
    n_modes = len(means)

    if sigmas is None:
        sigmas = np.ones(n_modes)
    else:
        sigmas = np.asarray(sigmas, dtype=float)

    if weights is None:
        weights = np.ones(n_modes) / n_modes
    else:
        weights = np.asarray(weights, dtype=float)

    components = rng.choice(n_modes, size=n_samples, p=weights)
    samples = rng.normal(loc=means[components], scale=sigmas[components])

    if return_components:
        return samples, components
    return samples


def make_grid(data, n_grid=1000, margin=0.0):
    """Create an evaluation grid for KDE/mode counting."""
    lo = float(np.min(data))
    hi = float(np.max(data))
    span = hi - lo
    return np.linspace(lo - margin * span, hi + margin * span, n_grid)


def scenario_names():
    """List preset simulation scenarios for Silverman-style illustrations."""
    return [
        "well_separated_bimodal",
        "well_separated_trimodal",
        "close_bimodal",
        "shoulder",
        "small_secondary_mode",
        "heteroskedastic_bimodal",
        "contaminated_unimodal",
        "heavy_tailed_unimodal",
        "skew_unimodal",
        "tiny_sample_bimodal",
    ]


def make_scenario(name, n_samples=300, rng=None):
    """Generate a simple preset dataset and metadata for illustration."""
    if rng is None:
        rng = np.random.default_rng()

    if name == "well_separated_bimodal":
        x = gen_mixture(
            n_samples,
            means=[-2.5, 2.5],
            sigmas=[0.7, 0.7],
            weights=[0.5, 0.5],
            rng=rng,
        )
        note = "Clear, separated modes. This is where the test usually looks good."
        expected_modes = 2
    elif name == "well_separated_trimodal":
        x = gen_mixture(
            n_samples,
            means=[-4.0, 0.0, 4.0],
            sigmas=[0.55, 0.65, 0.55],
            weights=[0.3, 0.4, 0.3],
            rng=rng,
        )
        note = "Another friendly case: distinct peaks with moderate sample size."
        expected_modes = 3
    elif name == "close_bimodal":
        x = gen_mixture(
            n_samples,
            means=[-0.9, 0.9],
            sigmas=[0.95, 0.95],
            weights=[0.5, 0.5],
            rng=rng,
        )
        note = "Two components are present, but separation is weak. Distinguishing 1 vs 2 modes gets hard."
        expected_modes = 2
    elif name == "shoulder":
        x = gen_mixture(
            n_samples,
            means=[0.0, 2.1],
            sigmas=[1.0, 0.4],
            weights=[0.88, 0.12],
            rng=rng,
        )
        note = "Classic shoulder case. The second component often looks like a bump rather than a full mode."
        expected_modes = 2
    elif name == "small_secondary_mode":
        x = gen_mixture(
            n_samples,
            means=[0.0, 3.2],
            sigmas=[0.9, 0.45],
            weights=[0.94, 0.06],
            rng=rng,
        )
        note = "A tiny second cluster can be scientifically meaningful but easy to smooth away."
        expected_modes = 2
    elif name == "heteroskedastic_bimodal":
        x = gen_mixture(
            n_samples,
            means=[-2.0, 1.8],
            sigmas=[1.4, 0.35],
            weights=[0.7, 0.3],
            rng=rng,
        )
        note = "Different scales across components can create one obvious peak and one broad hill."
        expected_modes = 2
    elif name == "contaminated_unimodal":
        base = rng.normal(0.0, 1.0, int(0.96 * n_samples))
        contam = rng.normal(5.5, 0.2, n_samples - len(base))
        x = np.concatenate([base, contam])
        note = "Mostly unimodal with a few outliers. The test can interpret contamination as an extra mode."
        expected_modes = 1
    elif name == "heavy_tailed_unimodal":
        x = rng.standard_t(df=3, size=n_samples)
        note = "Heavy-tailed but still unimodal. Good for showing that non-Gaussian shape does not imply multiple modes."
        expected_modes = 1
    elif name == "skew_unimodal":
        x = rng.lognormal(mean=0.2, sigma=0.55, size=n_samples)
        note = "Skewness alone is not multimodality, but rough samples can create spurious bumps."
        expected_modes = 1
    elif name == "tiny_sample_bimodal":
        x = gen_mixture(
            max(40, n_samples // 5),
            means=[-2.2, 2.2],
            sigmas=[0.8, 0.8],
            weights=[0.5, 0.5],
            rng=rng,
        )
        note = "Structure exists, but the sample is deliberately small. Sampling noise dominates."
        expected_modes = 2
    else:
        raise ValueError(f"Unknown scenario: {name}")

    return {
        "name": name,
        "data": np.asarray(x, dtype=float),
        "expected_modes": int(expected_modes),
        "note": note,
    }


def summarize_mode_scan(data, hs, x_grid=None):
    """Count KDE modes over a sequence of bandwidths."""
    data = np.asarray(data, dtype=float)
    hs = np.asarray(hs, dtype=float)
    if x_grid is None:
        x_grid = make_grid(data)

    n_modes = np.array([count_modes(data, x_grid, h) for h in hs], dtype=int)
    return {
        "hs": hs,
        "n_modes": n_modes,
        "x_grid": x_grid,
    }


def kde_values(data, x_grid, h):
    """Evaluate Gaussian KDE with bandwidth factor h on x_grid."""
    kde = gaussian_kde(data, bw_method=h)
    return kde(x_grid)


def count_modes(data, x_grid, h):
    """Count local maxima (modes) of KDE(data, h)."""
    y = kde_values(data, x_grid, h)
    peaks, _ = find_peaks(y)
    return len(peaks)


def search_h_critical(data, k_modes, x_grid=None, max_iter=60, tol=1e-8, h_low_factor=1e-6):
    """Binary search for critical bandwidth where KDE has <= k_modes.

    Returns a dict with final h_crit and full search history.
    """
    data = np.asarray(data, dtype=float)
    if x_grid is None:
        x_grid = make_grid(data)

    s = np.std(data)
    h_low = h_low_factor * s
    h_high = float(np.max(data) - np.min(data))

    while count_modes(data, x_grid, h_high) > k_modes:
        h_high *= 2.0

    history = []
    for _ in range(max_iter):
        h_mid = 0.5 * (h_low + h_high)
        k_now = count_modes(data, x_grid, h_mid)

        if k_now > k_modes:
            h_low = h_mid
        else:
            h_high = h_mid

        history.append(h_mid)
        if abs(h_high - h_low) < tol:
            break

    return {
        "h_crit": float(history[-1]),
        "history": history,
        "x_grid": x_grid,
        "k_modes_target": int(k_modes),
    }


def bootstrap_resample(data, rng=None):
    """Simple bootstrap resample of a 1D sample."""
    if rng is None:
        rng = np.random.default_rng()
    data = np.asarray(data)
    idx = rng.integers(0, len(data), size=len(data))
    return data[idx]


def _bootstrap_metric_once(data, k_modes, variant_fn, metric_key, variant_kwargs, seed):
    """Run a single bootstrap metric evaluation.

    This helper is defined at module scope so it can be used with process pools.
    """
    rng = np.random.default_rng(seed)
    sample = bootstrap_resample(data, rng=rng)
    out = variant_fn(sample, k_modes, **variant_kwargs)
    return float(out[metric_key])


def run_bootstrap_experiment(data, k_modes, n_boot=100, rng=None, search_fn=search_h_critical, **search_kwargs):
    """Run repeated bootstrap searches.

    search_fn must have signature compatible with:
      search_fn(sample, k_modes, **search_kwargs) -> dict with key 'h_crit'
    """
    if rng is None:
        rng = np.random.default_rng()

    h_values = []
    for _ in range(n_boot):
        sample = bootstrap_resample(data, rng=rng)
        out = search_fn(sample, k_modes, **search_kwargs)
        h_values.append(out["h_crit"])

    h_values = np.asarray(h_values)
    return {
        "h_values": h_values,
        "mean": float(np.mean(h_values)),
        "std": float(np.std(h_values, ddof=1)) if len(h_values) > 1 else 0.0,
        "q05": float(np.quantile(h_values, 0.05)),
        "q50": float(np.quantile(h_values, 0.50)),
        "q95": float(np.quantile(h_values, 0.95)),
    }


def compare_search_methods(data, k_modes, methods, common_kwargs=None):
    """Compare multiple search variants on same data.

    methods: dict like {"base": fn1, "variant_a": fn2}
    each fn should return dict with key 'h_crit'.
    """
    if common_kwargs is None:
        common_kwargs = {}

    results = {}
    for name, fn in methods.items():
        out = fn(data, k_modes, **common_kwargs)
        results[name] = out
    return results


def run_variant(data, k_modes, variant_fn, **variant_kwargs):
    """Run one variant function and return its result dict.

    Expected pattern:
      variant_fn(data, k_modes, **variant_kwargs) -> dict
    Example output keys can be: h_crit, p_value, x_star, ...
    """
    return variant_fn(data, k_modes, **variant_kwargs)


def run_bootstrap_metric(
    data,
    k_modes,
    variant_fn,
    metric_key,
    n_boot=100,
    rng=None,
    parallel=False,
    n_jobs=None,
    parallel_backend="thread",
    show_progress=True,
    **variant_kwargs,
):
    """Bootstrap any scalar metric produced by variant_fn.

    variant_fn should return a dict that contains metric_key.

    Parameters
    ----------
    parallel : bool, optional
        If True, evaluate bootstrap draws in parallel.
    n_jobs : int or None, optional
        Number of workers to use when parallel=True. None uses the executor
        default.
    parallel_backend : {"thread", "process"}, optional
        Executor backend to use for parallel execution. The default "thread"
        is safer in notebooks and marimo because it does not require
        variant_fn to be importable by child processes.
    show_progress : bool, optional
        If True, show a tqdm progress bar.
    """
    if rng is None:
        rng = np.random.default_rng()

    seeds = rng.integers(0, np.iinfo(np.uint32).max, size=n_boot, dtype=np.uint32)
    metric_values = []

    iterator = range(n_boot)
    if show_progress:
        iterator = tqdm(iterator, total=n_boot, desc=f"Bootstrap {metric_key}")
    for idx in iterator:
        metric_values.append(
            _bootstrap_metric_once(
                data=data,
                k_modes=k_modes,
                variant_fn=variant_fn,
                metric_key=metric_key,
                variant_kwargs=variant_kwargs,
                seed=int(seeds[idx]),
            )
        )

    metric_values = np.asarray(metric_values)
    return {
        "metric_key": metric_key,
        "values": metric_values,
        "mean": float(np.mean(metric_values)),
        "std": float(np.std(metric_values, ddof=1)) if len(metric_values) > 1 else 0.0,
        "q05": float(np.quantile(metric_values, 0.05)),
        "q50": float(np.quantile(metric_values, 0.50)),
        "q95": float(np.quantile(metric_values, 0.95)),
    }
