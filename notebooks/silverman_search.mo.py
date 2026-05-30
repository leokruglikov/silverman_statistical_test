import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import hvplot
    import hvplot.polars
    import polars as pl
    from scipy import stats
    from scipy.signal import find_peaks
    from scipy.stats import gaussian_kde
    import holoviews as hv

    return find_peaks, gaussian_kde, hv, mo, np, pl


@app.cell(hide_code=True)
def _(np):
    def gen_mixture( N_samples, means, sigmas=None, weights=None, *, rng=None, return_components=False ):

        if rng is None:
            rng = np.random.default_rng()

        means = np.asarray(means)

        if means.ndim == 0:
            raise ValueError("means must contain at least one component")

        N_modes = means.shape[0]

        if sigmas is None:
            sigmas = np.ones_like(means, dtype=float)
        else:
            sigmas = np.asarray(sigmas, dtype=float)

        if sigmas.shape != means.shape:
            raise ValueError(
                f"sigmas must have the same shape as means. "
                f"Got sigmas.shape={sigmas.shape}, means.shape={means.shape}"
            )

        if np.any(sigmas <= 0):
            raise ValueError("All sigmas must be positive")

        if weights is None:
            weights = np.ones(N_modes) / N_modes
        else:
            weights = np.asarray(weights, dtype=float)

        if weights.shape != (N_modes,):
            raise ValueError(
                f"weights must have shape ({N_modes},), got {weights.shape}"
            )

        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")

        if not np.isclose(weights.sum(), 1.0):
            raise ValueError("weights must sum to 1")

        components = rng.choice(N_modes, size=N_samples, p=weights)

        samples = rng.normal(
            loc=means[components],
            scale=sigmas[components]
        )

        if return_components:
            return samples, components

        return samples


    return (gen_mixture,)


@app.cell
def _(gen_mixture, pl):
    mixed_gauss_data = gen_mixture(
        N_samples=10000,
        means=[0,4,6,8],sigmas=[1.5, 1.0, 0.5, 0.6]
    )

    df_gauss = pl.DataFrame({'gmm_raw': mixed_gauss_data})
    df_gauss.hvplot.hist(y='gmm_raw', bins=200)
    return (mixed_gauss_data,)


@app.cell
def _(find_peaks, gaussian_kde, mixed_gauss_data, np):
    def get_n_modes(x,y,h):
        kde = gaussian_kde(y, bw_method=h)
        kde_y = kde(x)
        peaks, _ = find_peaks(kde_y)
        return len(peaks)

    K_MODES = 4
    max_iter_binary = 100
    tolerance = 1e-8
    n_grid = 1000
    x_grid = np.linspace(mixed_gauss_data.min(), mixed_gauss_data.max(), n_grid)
    s = np.std(mixed_gauss_data)
    h_low = 1e-6*s
    h_high = mixed_gauss_data.max() - mixed_gauss_data.min()

    while get_n_modes(x_grid, mixed_gauss_data, h_high) > K_MODES:
        h_high *= 2.0

    h_hist = []
    for i_ in range(max_iter_binary):
        h_mid = (h_low + h_high)/2.0
        k_proposed = get_n_modes(x=x_grid,y=mixed_gauss_data,h=h_mid)

        if k_proposed > K_MODES:
            # more modes than needed => more smoothing => higher h
            h_low = h_mid
        else:
            # less modes than needed => less smoothing => lower h
            h_high = h_mid

        h_hist.append(h_mid)

        if np.abs(h_high - h_low) < tolerance:
            break
    return h_hist, x_grid


@app.cell
def _(gaussian_kde, h_hist):
    def get_kde_xy(x, y, h):
        kde = gaussian_kde(y, bw_method=h)
        kde_y = kde(x)
        return x, kde_y

    h_critical = float(h_hist[-1])
    return get_kde_xy, h_critical


@app.cell(hide_code=True)
def _(h_critical, mo):
    h_slider = mo.ui.slider(
        start=max(h_critical * 0.2, 1e-6),
        stop=h_critical * 4,
        step=max(h_critical * 0.005, 1e-6),
        value=h_critical,
        label="h",
        show_value=True,
    )
    h_slider
    return (h_slider,)


@app.cell
def _(
    find_peaks,
    get_kde_xy,
    h_critical,
    h_slider,
    hv,
    mixed_gauss_data,
    mo,
    x_grid,
):
    x_kde_, y_kde_ = get_kde_xy(x_grid, mixed_gauss_data, h_slider.value)
    peak_idx, _ = find_peaks(y_kde_)
    peak_x = x_kde_[peak_idx]
    peak_y = y_kde_[peak_idx]

    kde_curve = hv.Curve(
        (x_kde_, y_kde_),
        kdims="x",
        vdims="density",
    ).opts(
        width=900,
        height=420,
        line_width=3,
        color="#0f766e",
        xlabel="x",
        ylabel="density",
        toolbar="above",
    )

    peaks = hv.Scatter(
        (peak_x, peak_y),
        kdims="x",
        vdims="density",
    ).opts(
        color="#dc2626",
        size=8,
        marker="diamond",
    )

    summary = mo.md(
        f"`h = {h_slider.value:.6f}`    `h_crit = {h_critical:.6f}`    `peaks = {len(peak_idx)}`"
    )

    mo.vstack(
        [
            h_slider,
            summary,
            (kde_curve * peaks),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
