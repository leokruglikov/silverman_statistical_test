import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


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

    return find_peaks, gaussian_kde, hv, np, pl


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


@app.cell(hide_code=True)
def _(gen_mixture, pl):
    mixed_gauss_data = gen_mixture(
        N_samples=10000,
        means=[0,4,6,8],sigmas=[2, 1.0, 0.5, 0.6]
    )

    df_gauss = pl.DataFrame({'gmm_raw': mixed_gauss_data})
    df_gauss.hvplot.hist(y='gmm_raw', bins=200)
    return (mixed_gauss_data,)


@app.cell
def _(find_peaks, gaussian_kde, hv, mixed_gauss_data, np):
    h_start = 0.001
    h_end = 0.1
    n_steps = 40
    hs = np.linspace(h_start, h_end, n_steps)
    N_grid = 1000

    data = mixed_gauss_data
    frequencies_base, edges_base = np.histogram(data,bins='scott')
    x = np.linspace(
        data.min(), data.max(), N_grid
    )

    n_peaks_all = []
    for h in hs:
        kde = gaussian_kde(data, bw_method=h)
        peak, _ = find_peaks(kde(x))
        n_peaks_all.append( len(peak) )
    
    
    
    hv.Scatter((hs, np.array(n_peaks_all))).opts(height=600, width=800)+\
    hv.Scatter((hs[1:], np.diff( np.array(n_peaks_all)))).opts(height=600, width=800)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
