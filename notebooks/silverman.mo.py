import marimo

__generated_with = "0.23.6"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _():
    import marimo as mo
    import holoviews as hv
    import numpy as np
    from holoviews import opts
    opts.defaults(
        opts.Curve(width=800, height=400, color='red'),
        opts.Histogram(width=800, height=400, fontscale=1.5),
        opts.Scatter(width=800, height=400, size=10, color='green'),
        opts.Image(width=800, height=500),
    )
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks

    return find_peaks, gaussian_kde, hv, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generate the "test" distribution
    The function `gen_mixture()` generates a Gaussian mixture distribution
    """)
    return


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

    gaussian_mixture =  gen_mixture(
        N_samples=1000,
        means=[0,4,6,8],sigmas=[1.5, 1.0, 0.5, 0.6]
        )
    return (gaussian_mixture,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Specify the parameters
    Specify the parameters for the binary search of the threshold bandwidth $h^{\text{crit}}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    k_modes_ui = mo.ui.number(value=4)
    max_iter_binary_ui = mo.ui.number(value=10)
    tolerance_ui = mo.ui.number(value=1e-8)
    n_grid_ui = mo.ui.number(value=1000)
    binary_search_ui = mo.ui.dictionary(
        {
            "kmodes": k_modes_ui,
            "max_iters": max_iter_binary_ui,
            "tolerance": tolerance_ui,
            "n_grid": n_grid_ui
        }
    )
    binary_search_ui
    return (binary_search_ui,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Binary search function
    The function `search_binary_h()` performs the binary search of $h^{\text{crit}}$ and returns the full "history" of the bandwidth's that the search went over.
    """)
    return


@app.cell(hide_code=True)
def _(find_peaks, gaussian_kde, np):
    def search_binary_h(mixed_gauss_data, x_grid, k_modes, max_iter_binary, tolerance, h_low_factor=1e-6):
        s = np.std(mixed_gauss_data)
        h_low = 1e-6*s
        h_high = mixed_gauss_data.max() - mixed_gauss_data.min()

        def get_n_modes(x,y,h):
            kde = gaussian_kde(y, bw_method=h)
            kde_y = kde(x)
            peaks, _ = find_peaks(kde_y)
            return len(peaks)

        while get_n_modes(x_grid, mixed_gauss_data, h_high) > k_modes:
            h_high *= 2.0

        h_hist = []
        for i_ in range(max_iter_binary):
            h_mid = (h_low + h_high)/2.0
            k_proposed = get_n_modes(x=x_grid,y=mixed_gauss_data,h=h_mid)

            if k_proposed > k_modes:
                # more modes than needed => more smoothing => higher h
                h_low = h_mid
            else:
                # less modes than needed => less smoothing => lower h
                h_high = h_mid

            h_hist.append(h_mid)

            if np.abs(h_high - h_low) < tolerance:
                break
        return h_hist


    return (search_binary_h,)


@app.cell
def _(binary_search_ui, gaussian_mixture, np, search_binary_h):
    k_modes = binary_search_ui.value['kmodes']
    max_iters_binary = binary_search_ui.value['max_iters']
    tolerance = binary_search_ui.value['tolerance']
    n_grid = binary_search_ui.value['n_grid']
    x_grid = np.linspace(gaussian_mixture.min(), gaussian_mixture.max(), n_grid)
    h_crit_history = search_binary_h(
        gaussian_mixture, x_grid, k_modes, max_iters_binary, tolerance
    )
    return h_crit_history, k_modes, max_iters_binary, tolerance, x_grid


@app.cell(hide_code=True)
def _(hv, np):
    def plot_histogram(data):
        freq, edges = np.histogram(data, bins='scott', density=True)
        return hv.Histogram((edges, freq))

    def gauss_kde_plot(data, x_grid, h, show_peaks=True):
        from scipy.stats import gaussian_kde
        from scipy.signal import find_peaks
        import holoviews as hv

        kde = gaussian_kde(data, bw_method=h)
        y = kde(x_grid)
        peaks, _ = find_peaks(y)

        curve = hv.Curve((x_grid, y))
        if show_peaks and len(peaks) > 0:
            return curve * hv.Scatter((x_grid[peaks], y[peaks]))
        return curve



    return gauss_kde_plot, plot_histogram


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualize the history
    The critical bandwidth's history can be visualized.
    """)
    return


@app.cell
def _(max_iters_binary, mo):
    n_iter_slider = mo.ui.slider(start=1, stop=max_iters_binary, step=1, value=max_iters_binary,full_width=True,label='Iteration of h_critical')
    n_iter_slider
    return (n_iter_slider,)


@app.cell
def _(
    gauss_kde_plot,
    gaussian_mixture,
    h_crit_history,
    n_iter_slider,
    plot_histogram,
    x_grid,
):
    histogram_plot = plot_histogram( 
        gaussian_mixture
    )
    histogram_kde_plot = gauss_kde_plot(gaussian_mixture,x_grid,h=h_crit_history[n_iter_slider.value-1])
    histogram_plot*histogram_kde_plot
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ### Parameters for bootstrap
    Specify the parameters for the bootstrapping method.
    """)
    return


@app.cell
def _(mo):
    b_bootstrap_ui = mo.ui.number(value=10)
    bootstrap_dict_ui = mo.ui.dictionary(
        {
            "b_bootstrap": b_bootstrap_ui
        }
    )
    bootstrap_dict_ui
    return (bootstrap_dict_ui,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bootstrap
    Run the bootstrap iteration and collect the obtained $h^{\text{crit}}$. The same function `search_binary_h()` is used. At the end, compute the p-value using the indicator function.
    """)
    return


@app.cell(hide_code=True)
def _(
    bootstrap_dict_ui,
    gaussian_mixture,
    h_crit_history,
    k_modes,
    max_iters_binary,
    mo,
    np,
    search_binary_h,
    tolerance,
    x_grid,
):
    from joblib import Parallel, delayed

    def generate_xstar( x, h_obs, noise_scale=1.0, variance_corrected=False, ddof=1, rng=None, ):
        x = np.asarray(x, dtype=float)
        n = len(x)

        if rng is None:
            rng = np.random.default_rng()

        bootstrap_random_indices = rng.choice(n, size=n, replace=True)
        base = x[bootstrap_random_indices]

        random_normal_zs = rng.normal(loc=0.0, scale=1.0, size=n)
        smooth_noise = noise_scale * h_obs * random_normal_zs
        if not variance_corrected:
            x_star = base + smooth_noise
        else:
            x_mean = np.mean(x)
            x_std = np.std(x, ddof=ddof)

            if x_std <= 0:
                raise ValueError("Division by 0")

            correction = np.sqrt(1.0 + (noise_scale * h_obs) ** 2 / x_std**2)

            x_star = x_mean + ((base - x_mean) + smooth_noise) / correction

        return x_star


    b_bootstrap = bootstrap_dict_ui.value['b_bootstrap']
    boot_id = 0

    def run_bootstrap_iteration(gaussian_mixture, h_crit_last, x_grid, k_modes, max_iters_binary, tolerance):
        x_star = generate_xstar(gaussian_mixture, h_crit_last)
        res = search_binary_h(
            mixed_gauss_data=x_star,
            x_grid=x_grid,
            k_modes=k_modes,
            max_iter_binary=max_iters_binary,
            tolerance=tolerance
        )
        return res[-1]

    h_crit_last = h_crit_history[-1]
    executor = Parallel(n_jobs=-1, return_as="generator")

    tasks = (
        delayed(run_bootstrap_iteration)(
            gaussian_mixture, h_crit_last, x_grid, k_modes, max_iters_binary, tolerance
        )
        for _ in range(b_bootstrap)
    )

    h_stars = list(mo.status.progress_bar(executor(tasks), total=b_bootstrap))

    p_val = np.mean(h_stars > h_crit_history[-1])
    return (p_val,)


@app.cell(hide_code=True)
def _(mo, p_val):
    mo.md(r"""
     ### Interpreting the p-value
     - A high p-value (e.g. $p>0.05$ for $0.95$ confidence) suggests evidence for $H_0 \rightarrow$ $k$ or less modes.
     - A low p-value (e.g. $p<0.05$) suggests evidence for rejecting $H_0 \rightarrow$ in favor of the alternative $H_1 \rightarrow$ more than $k$ modes. 

     In this case: $p=${}
    """.format(p_val))
    return


@app.cell
def _(p_val):
    p_val
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
