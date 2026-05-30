import marimo

__generated_with = "0.23.6"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    import numpy as np
    import holoviews as hv
    from holoviews import opts
    from scipy.signal import find_peaks
    from silverman import utils as su
    opts.defaults(
        opts.Curve(width=800, height=400, color='red'),
        opts.Histogram(width=800, height=400, fontscale=1.5),
        opts.Scatter(width=800, height=400, size=10, color='green'),
        opts.Image(width=800, height=500),
    )
    from joblib import Parallel, delayed
    from wigglystuff import EnvConfig

    return Parallel, delayed, hv, mo, np, su


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Silverman test playground
    This notebook shows how to use `silverman_utils.py` for experiments.

    Pattern:
    1. Define a variant function returning a dict (`h_crit`, `x_star`, `p_value`, ...)
    2. Run it once with `su.run_variant(...)`
    3. Bootstrap any metric with `su.run_bootstrap_metric(...)`
    """)
    return


@app.cell(hide_code=True)
def global_config(mo):
    GLOBAL_CONFIG = mo.ui.dictionary(
        {
            "n_samples_gaussian_mixture": mo.ui.number(
                value=1500,
                start=100,
                stop=10_000,
                step=100,
                label="Number of samples in the Gaussian mixture",
            ),

            "k_modes": mo.ui.number(
                value=4,
                start=1,
                stop=10,
                step=1,
                label="Number of modes tested",
            ),

            "n_grid": mo.ui.number(
                value=1000,
                start=100,
                stop=10_000,
                step=100,
                label="Number of points in the grid for KDE evaluation",
            ),

            "max_iters": mo.ui.number(
                value=15,
                start=1,
                stop=50,
                step=1,
                label="Maximum iterations for h_critical search",
            ),
            "tol": mo.ui.number(
                value=1e-6,
                start=1e-10,
                stop=1e-2,
                step=1e-6,
                label="Tolerance for h_critical search convergence",
            ),
            "b_boot": mo.ui.number(
                value=50,
                start=10,
                stop=1000,
                step=10,
                label="Number of bootstrap samples in the Silverman's algorithm",
            ),
            "n_boot": mo.ui.number(
                value=100,
                start=10,
                stop=1000,
                step=10,
                label="Number of bootstrap runs for estimating the p-value (or other metric)",
            ),
        }
    ).form(submit_button_label="Config OK")
    GLOBAL_CONFIG
    return (GLOBAL_CONFIG,)


@app.cell
def _(GLOBAL_CONFIG, su):
    gaussian_mixture = su.gen_mixture(
        n_samples=GLOBAL_CONFIG.value['n_samples_gaussian_mixture'],
        means=[-1, 4],
        sigmas=[0.8, 1.0],
    )
    return (gaussian_mixture,)


@app.cell
def _(GLOBAL_CONFIG, gaussian_mixture, hv, np, save, su):
    x_grid = su.make_grid(gaussian_mixture, n_grid=GLOBAL_CONFIG.value['n_grid'])
    y = su.kde_values(gaussian_mixture, x_grid, h=0.12)
    peaks, _ = __import__("scipy.signal", fromlist=["find_peaks"]).find_peaks(y)

    hist = hv.Histogram(np.histogram(gaussian_mixture, bins="scott", density=True))
    kde = hv.Curve((x_grid, y))
    pts = hv.Scatter((x_grid[peaks], y[peaks]))

    hist * (kde * pts)
    save(hist * (kde * pts), filename='bimodal_simple_distribution.html')
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plain (baseline) Silverman-style variant
    """)
    return


@app.cell(hide_code=True)
def _(Parallel, delayed, np, su):
    def _silverman_boot_one(data, k_modes, h_crit, seed, search_kwargs):
        rng = np.random.default_rng(seed)

        xb = (
            su.bootstrap_resample(data, rng=rng)
            + rng.normal(0.0, h_crit, size=len(data))
        )

        out_b = su.search_h_critical(xb, k_modes, **search_kwargs)
        return out_b["h_crit"]


    def plain_silverman_variant_parallel(
        data,
        k_modes,
        b_boot=80,
        rng=None,
        n_jobs=-1,
        **search_kwargs
    ):
        if rng is None:
            rng = np.random.default_rng()

        data = np.asarray(data)

        base = su.search_h_critical(data, k_modes, **search_kwargs)
        h_crit = base["h_crit"]

        seeds = rng.integers(
            0,
            np.iinfo(np.uint32).max,
            size=b_boot,
            dtype=np.uint32,
        )

        h_stars = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_silverman_boot_one)(
                data,
                k_modes,
                h_crit,
                int(seed),
                search_kwargs,
            )
            for seed in seeds
        )

        h_stars = np.asarray(h_stars)
        p_value = float(np.mean(h_stars >= h_crit))

        return {
            "h_crit": h_crit,
            "p_value": p_value,
            "h_stars": h_stars,
            "search_history": base["history"],
        }

    return (plain_silverman_variant_parallel,)


@app.cell
def _(GLOBAL_CONFIG, gaussian_mixture, plain_silverman_variant_parallel, su):
    plain_silverman_run = su.run_variant(
        gaussian_mixture,
        k_modes=GLOBAL_CONFIG.value['k_modes'],
        variant_fn=plain_silverman_variant_parallel,
        b_boot=GLOBAL_CONFIG.value['b_boot'],
        max_iter=GLOBAL_CONFIG.value['max_iters'],
        tol=GLOBAL_CONFIG.value['tol'],
    )
    return


@app.cell
def _(mo):
    run_button_plain = mo.ui.run_button(label="Run plain Silverman bootstrap")
    run_button_plain
    return (run_button_plain,)


@app.cell
def _(
    GLOBAL_CONFIG,
    gaussian_mixture,
    mo,
    plain_silverman_variant_parallel,
    run_button_plain,
    su,
):
    mo.stop(not run_button_plain.value, mo.md("Click the button to run this plain Silverman bootstrap"))

    boot_p_plain = su.run_bootstrap_metric(
        gaussian_mixture,
        k_modes=GLOBAL_CONFIG.value['k_modes'],
        variant_fn=plain_silverman_variant_parallel,
        metric_key="p_value",
        parallel=True,
        show_progress=True,
        n_boot=GLOBAL_CONFIG.value['n_boot'],
        b_boot=GLOBAL_CONFIG.value['b_boot'],
        max_iter=GLOBAL_CONFIG.value['max_iters']
    )
    return (boot_p_plain,)


@app.cell
def _(boot_p_plain, hv, np):
    from holoviews import save
    freq_plain, edges_plain = np.histogram(boot_p_plain['values'], bins=10, density=True)

    pval_hist_plain = (
        hv.Histogram((edges_plain, freq_plain)).opts(title='p-values bootstrapped')*
        hv.VLines([boot_p_plain['values'].mean()]).opts(color='red')*
        hv.Distribution(boot_p_plain['values'], label='Bootstrap p-values').opts(color='blue', alpha=0.1)
    ).opts(tools=['hover'])
    #save(pval_hist_plain, filename='pval_plain_silverman_1.html')
    pval_hist_plain
    return pval_hist_plain, save


@app.cell
def _(boot_p_plain, mo):
    mo.md(f"""
    Bootstrap of `p_value` across repeated runs of your variant:
    ,lo

    `mean={boot_p_plain['mean']:.4f}`  
    `q05={boot_p_plain['q05']:.4f}`  
    `q50={boot_p_plain['q50']:.4f}`  
    `q95={boot_p_plain['q95']:.4f}`
    """)
    return


@app.cell
def _(hv, pval_hist_plain):
    hv.save(pval_hist_plain, filename='pval_hist_plain.html',fmt='html', backend='bokeh')
    return


@app.cell
def _():
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ### Silverman with scaled variance
    """)
    return


@app.cell(hide_code=True)
def _(Parallel, delayed, np, su):
    def _silverman_boot_one_corrected_var(
        data,
        k_modes,
        h_crit,
        seed,
        search_kwargs,
    ):
        rng = np.random.default_rng(seed)

        data = np.asarray(data, dtype=float)
        n = len(data)

        base = su.bootstrap_resample(data, rng=rng)
        noise = rng.normal(0.0, h_crit, size=n)

        x_mean = np.mean(data)
        x_std = np.std(data, ddof=1)

        if x_std <= 0:
            raise ValueError("Cannot use variance correction when data variance is zero.")

        correction = np.sqrt(1.0 + h_crit**2 / x_std**2)

        xb = x_mean + ((base - x_mean) + noise) / correction

        out_b = su.search_h_critical(xb, k_modes, **search_kwargs)
        return out_b["h_crit"]

    def corrected_var_silverman_variant_parallel(
        data,
        k_modes,
        b_boot=80,
        rng=None,
        n_jobs=-1,
        **search_kwargs
    ):
        if rng is None:
            rng = np.random.default_rng()

        data = np.asarray(data)

        base = su.search_h_critical(data, k_modes, **search_kwargs)
        h_crit = base["h_crit"]

        seeds = rng.integers(
            0,
            np.iinfo(np.uint32).max,
            size=b_boot,
            dtype=np.uint32,
        )

        h_stars = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_silverman_boot_one_corrected_var)(
                data,
                k_modes,
                h_crit,
                int(seed),
                search_kwargs,
            )
            for seed in seeds
        )

        h_stars = np.asarray(h_stars)
        p_value = float(np.mean(h_stars >= h_crit))

        return {
            "h_crit": h_crit,
            "p_value": p_value,
            "h_stars": h_stars,
            "search_history": base["history"],
        }


    return (corrected_var_silverman_variant_parallel,)


@app.cell
def _(
    GLOBAL_CONFIG,
    corrected_var_silverman_variant_parallel,
    gaussian_mixture,
    su,
):
    corrected_var_silverman_run = su.run_variant(
        gaussian_mixture,
        k_modes=GLOBAL_CONFIG.value['k_modes'],
        variant_fn= corrected_var_silverman_variant_parallel,
        b_boot=GLOBAL_CONFIG.value['b_boot'],
        max_iter=GLOBAL_CONFIG.value['max_iters'],
        tol=GLOBAL_CONFIG.value['tol'],
    )
    return


@app.cell
def _(mo):
    run_button_corrected_var = mo.ui.run_button(label="Run Silverman bootstrap with corrected variance")
    run_button_corrected_var
    return (run_button_corrected_var,)


@app.cell
def _(
    GLOBAL_CONFIG,
    corrected_var_silverman_variant_parallel,
    gaussian_mixture,
    mo,
    run_button_corrected_var,
    su,
):
    mo.stop(not run_button_corrected_var.value, mo.md("Click the button to run the plain Silverman bootstrap before running the corrected variance version."))
    boot_p_corrected_var = su.run_bootstrap_metric(
        gaussian_mixture,
        k_modes=GLOBAL_CONFIG.value['k_modes'],
        variant_fn=corrected_var_silverman_variant_parallel,
        metric_key="p_value",
        parallel=True,
        show_progress=True,
        n_boot=GLOBAL_CONFIG.value['n_boot'],
        b_boot=GLOBAL_CONFIG.value['b_boot'],
        max_iter=GLOBAL_CONFIG.value['max_iters']
    )
    return (boot_p_corrected_var,)


@app.cell
def _(boot_p_corrected_var, hv, np, save):
    freq_corrected_var, edges_corrected_var = np.histogram(boot_p_corrected_var['values'], bins=10, density=True)
    #hv.Histogram((edges_corrected_var, freq_corrected_var)).opts(title='p-values bootstrapped (corrected variance)')
    pval_hist_correction_var = (
        hv.Histogram((edges_corrected_var, freq_corrected_var)).opts(title='p-values bootstrapped')*
        hv.VLines([boot_p_corrected_var['values'].mean()]).opts(color='red')*
        hv.Distribution(boot_p_corrected_var['values'], label='Bootstrap p-values (variance correction)').opts(color='blue', alpha=0.1)
    ).opts(tools=['hover'])
    save(pval_hist_correction_var, filename='pval_var_correction_simple_bimodal.html')
    return


@app.cell
def _(boot_p_corrected_var, boot_p_plain):
    boot_p_corrected_var['values'].mean()
    boot_p_plain['values'].mean()
    return


@app.cell
def _():
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""
    ### Silverman with Hall York correction
    """)
    return


@app.cell
def _(mo):
    alpha_level_selector = mo.ui.number(value=0.05, label="Select alpha for Hall York correction.\n Default 0.05 - a 95% confidence level.")
    alpha_level_selector
    return (alpha_level_selector,)


@app.cell(hide_code=True)
def _(Parallel, alpha_level_selector, delayed, np, su):
    def get_hall_york_lambda(alpha):
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
        return num / den

    def _silverman_boot_one_corrected_var_hall_york(
        data,
        k_modes,
        h_crit,
        seed,
        search_kwargs,
    ):
        rng = np.random.default_rng(seed)

        data = np.asarray(data, dtype=float)
        n = len(data)

        base = su.bootstrap_resample(data, rng=rng)
        noise = rng.normal(0.0, h_crit, size=n)

        x_mean = np.mean(data)
        x_std = np.std(data, ddof=1)

        if x_std <= 0:
            raise ValueError("Cannot use variance correction when data variance is zero.")

        correction = np.sqrt(1.0 + h_crit**2 / x_std**2)

        xb = x_mean + ((base - x_mean) + noise) / correction

        out_b = su.search_h_critical(xb, k_modes, **search_kwargs)
        return out_b["h_crit"]

    def corrected_var_silverman_hall_york_variant_parallel(
        data,
        k_modes,
        b_boot=80,
        rng=None,
        n_jobs=-1,
        **search_kwargs
    ):
        if rng is None:
            rng = np.random.default_rng()

        data = np.asarray(data)

        base = su.search_h_critical(data, k_modes, **search_kwargs)
        h_crit = base["h_crit"]

        seeds = rng.integers(
            0,
            np.iinfo(np.uint32).max,
            size=b_boot,
            dtype=np.uint32,
        )

        h_stars = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_silverman_boot_one_corrected_var_hall_york)(
                data,
                k_modes,
                h_crit,
                int(seed),
                search_kwargs,
            )
            for seed in seeds
        )

        h_stars = np.asarray(h_stars)
        lambda_hall_york = get_hall_york_lambda(alpha_level_selector.value)
        p_value = float(np.mean(h_stars >= h_crit*lambda_hall_york))

        return {
            "h_crit": h_crit,
            "p_value": p_value,
            "h_stars": h_stars,
            "search_history": base["history"],
        }

    return (corrected_var_silverman_hall_york_variant_parallel,)


@app.cell
def _(mo):
    run_button_hall_york = mo.ui.run_button(label="Run Silverman bootstrap with corrected variance + Hall York correction")
    run_button_hall_york
    return (run_button_hall_york,)


@app.cell
def _(
    GLOBAL_CONFIG,
    corrected_var_silverman_hall_york_variant_parallel,
    gaussian_mixture,
    su,
):
    corrected_var_hall_york_silverman_run = su.run_variant(
        gaussian_mixture,
        k_modes=GLOBAL_CONFIG.value['k_modes'],
        variant_fn=corrected_var_silverman_hall_york_variant_parallel,
        b_boot=GLOBAL_CONFIG.value['b_boot'],
        max_iter=GLOBAL_CONFIG.value['max_iters'],
        tol=GLOBAL_CONFIG.value['tol']
    )
    return


@app.cell
def _(
    GLOBAL_CONFIG,
    corrected_var_silverman_hall_york_variant_parallel,
    gaussian_mixture,
    mo,
    run_button_hall_york,
    su,
):
    mo.stop(not run_button_hall_york.value, mo.md("Click the button to run the Silverman bootstrap with corrected variance before running the version with Hall York correction."))
    boot_p_corrected_var_hall_york = su.run_bootstrap_metric(
        gaussian_mixture,
        k_modes=GLOBAL_CONFIG.value['k_modes'],
        variant_fn=corrected_var_silverman_hall_york_variant_parallel,
        metric_key="p_value",
        parallel=True,
        show_progress=True,
        n_boot=GLOBAL_CONFIG.value['n_boot'],
        b_boot=GLOBAL_CONFIG.value['b_boot'],
        max_iter=GLOBAL_CONFIG.value['max_iters']
    )
    return (boot_p_corrected_var_hall_york,)


@app.cell
def _(boot_p_corrected_var_hall_york, hv, np):
    freq_corrected_var_hall_york, edges_corrected_var_hall_york = np.histogram(boot_p_corrected_var_hall_york['values'], bins=10, density=True)
    hv.Histogram((edges_corrected_var_hall_york, freq_corrected_var_hall_york)).opts(title='p-values bootstrapped (corrected variance + Hall York)')

    pval_hist_correction_var_hy = (
        hv.Histogram((edges_corrected_var_hall_york, freq_corrected_var_hall_york)).opts(title='p-values bootstrapped')*
        hv.VLines([boot_p_corrected_var_hall_york['values'].mean()]).opts(color='red')*
        hv.Distribution(boot_p_corrected_var_hall_york['values'], label='Bootstrap p-values (variance + HY)').opts(color='blue', alpha=0.1)
    ).opts(tools=['hover'])

    pval_hist_correction_var_hy
    return


@app.cell
def _(boot_p_corrected_var_hall_york):
    boot_p_corrected_var_hall_york
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
