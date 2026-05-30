import marimo

__generated_with = "0.23.6"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo
    import holoviews as hv
    import numpy as np
    import silverman_utils as su
    from joblib import Parallel, delayed
    from holoviews import opts
    from scipy.signal import find_peaks

    opts.defaults(
        opts.Curve(width=850, height=380),
        opts.Histogram(width=850, height=380),
        opts.Scatter(width=850, height=380, size=8),
    )
    return Parallel, delayed, find_peaks, hv, mo, np, su


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Silverman: where it shines, where it breaks

    This notebook is a simple playground for illustrating Silverman-style mode testing.
    """)
    return


@app.cell(hide_code=True)
def _(mo, su):
    controls = mo.ui.dictionary(
        {
            "scenario": mo.ui.dropdown(
                options=su.scenario_names(),
                value="well_separated_bimodal",
                label="Scenario",
            ),
            "n_samples": mo.ui.number(value=300, start=50, stop=3000, step=50, label="n"),
            "seed": mo.ui.number(value=12, start=0, stop=9999, step=1, label="seed"),
            "k_modes_tested": mo.ui.number(value=1, start=1, stop=4, step=1, label="Test H0: modes <= k"),
            "n_grid": mo.ui.number(value=1000, start=200, stop=5000, step=100, label="KDE grid"),
            "b_boot": mo.ui.number(value=80, start=20, stop=400, step=20, label="Bootstrap draws"),
            "n_reps": mo.ui.number(value=24, start=5, stop=100, step=1, label="Repeated simulations"),
        }
    ).form(submit_button_label="Run")
    controls
    return (controls,)


@app.cell
def _(controls, np, su):
    cfg = controls.value
    scenario_rng = np.random.default_rng(cfg["seed"])
    scenario = su.make_scenario(cfg["scenario"], n_samples=cfg["n_samples"], rng=scenario_rng)
    x = scenario["data"]
    x_grid = su.make_grid(x, n_grid=cfg["n_grid"], margin=0.05)
    search = su.search_h_critical(x, cfg["k_modes_tested"], x_grid=x_grid)
    silverman = su.plain_silverman_variant(
        x,
        cfg["k_modes_tested"],
        b_boot=cfg["b_boot"],
        x_grid=x_grid,
    )
    return cfg, scenario, search, silverman, x, x_grid


@app.cell(hide_code=True)
def _(cfg, mo, scenario, search, silverman):
    mo.md(f"""
    ### Reading of the current case

    `{scenario["name"]}`  
    expected visible modes: `{scenario["expected_modes"]}`  
    tested null hypothesis: `modes <= {cfg["k_modes_tested"]}`  
    estimated critical bandwidth: `{search["h_crit"]:.4f}`  
    simple Silverman-style p-value: `{silverman["p_value"]:.4f}`

    {scenario["note"]}
    """)
    return


@app.cell
def _(find_peaks, hv, np, su, x):
    def plot_histogram(data):
        freq, edges = np.histogram(data, bins="scott", density=True)
        return hv.Histogram((edges, freq)).opts(color="#cbd5e1", line_color="#475569")

    def plot_kde(data, x_grid, h, color):
        y = su.kde_values(data, x_grid, h)
        peaks, _ = find_peaks(y)
        curve = hv.Curve((x_grid, y)).opts(color=color, line_width=3)
        pts = hv.Scatter((x_grid[peaks], y[peaks])).opts(color=color, marker="diamond")
        return curve * pts

    h_small = max(0.5 * np.std(x) / max(len(x) ** 0.2, 1.0), 1e-3)
    hist = plot_histogram(x)
    return h_small, hist, plot_kde


@app.cell
def _(h_small, hist, plot_kde, save, search, x, x_grid):
    save((
        hist
        * plot_kde(x, x_grid, h_small, "#2563eb")
        * plot_kde(x, x_grid, search["h_crit"], "#dc2626")
    ).opts(title="Blue: less smoothing, Red: critical bandwidth"), 'close_bimodal_distribution.html')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Advantages and non-advantages

    Weak cases:
    - weak separation or shoulder structure
    - tiny secondary groups
    - contamination and outliers
    - very small samples
    """)
    return


@app.cell
def _(hv, np, scenario, search, su, x_grid):
    from holoviews import save
    hs = np.linspace(max(search["h_crit"] * 0.15, 1e-3), search["h_crit"] * 2.2, 50)
    scan = su.summarize_mode_scan(scenario["data"], hs, x_grid=x_grid)
    mode_count_h_plot = hv.Curve((scan["hs"], scan["n_modes"])).opts(
        color="#0f766e",
        line_width=3,
        xlabel="bandwidth h",
        ylabel="number of KDE modes",
        title="Mode count across smoothing levels",
    )*\
    hv.Scatter((scan["hs"], scan["n_modes"])).opts(
        color="#0f766e",
        line_width=3,
        xlabel="bandwidth h",
        ylabel="number of KDE modes",
        title="Mode count across smoothing levels",
    )

    #save(mode_count_h_plot,filename='mode_counting_h_clear_bimodal.html')
    mode_count_h_plot
    return (save,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Repeated draws

    A single picture can be convincing for the wrong reason.
    Repeating the same scenario shows whether the conclusion is stable or fragile.
    """)
    return


@app.cell
def _(Parallel, cfg, delayed, np, scenario, su):
    def run_repeated_draw_parallel(rep_idx_parallel, cfg_parallel, scenario_parallel):
        repeated_rng_parallel = np.random.default_rng(cfg_parallel["seed"] + rep_idx_parallel)
        repeated_sample_parallel = su.make_scenario(
            scenario_parallel["name"],
            n_samples=cfg_parallel["n_samples"],
            rng=repeated_rng_parallel,
        )["data"]
        repeated_run_parallel = su.plain_silverman_variant(
            repeated_sample_parallel,
            cfg_parallel["k_modes_tested"],
            b_boot=cfg_parallel["b_boot"],
            max_iter=40,
            tol=1e-6,
        )
        return repeated_run_parallel["h_crit"], repeated_run_parallel["p_value"]

    repeated_results_parallel = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_repeated_draw_parallel)(rep_idx_parallel, cfg, scenario)
        for rep_idx_parallel in range(cfg["n_reps"])
    )

    h_crit_values_parallel = np.asarray([item[0] for item in repeated_results_parallel])
    p_values_parallel = np.asarray([item[1] for item in repeated_results_parallel])
    return h_crit_values_parallel, p_values_parallel


@app.cell
def _(h_crit_values_parallel, hv, np, p_values_parallel, save):
    p_freq, p_edges = np.histogram(p_values_parallel, bins=10, density=True)
    h_freq, h_edges = np.histogram(h_crit_values_parallel, bins=10, density=True)

    p_plot = hv.Histogram((p_edges, p_freq)).opts(
        color="#93c5fd",
        line_color="#1d4ed8",
        title="Repeated p-values",
    )
    h_plot = hv.Histogram((h_edges, h_freq)).opts(
        color="#fca5a5",
        line_color="#b91c1c",
        title="Repeated h_crit values",
        axiswise=True
    )
    p_plot + h_plot
    save(p_plot, 'close_bimodal_pvalue.html')
    p_plot + h_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sample-size sweep

    This gives a direct way to show where the method starts to stabilize and where it is too noisy.
    """)
    return


@app.cell
def _(Parallel, cfg, delayed, np, scenario, su):
    def run_sample_size_parallel(n_now_parallel, rep_idx_n_parallel, scenario_parallel, cfg_parallel):
        sample_size_rng_parallel = np.random.default_rng(
            10_000 + 100 * n_now_parallel + rep_idx_n_parallel
        )
        sample_size_sample_parallel = su.make_scenario(
            scenario_parallel["name"],
            n_samples=int(n_now_parallel),
            rng=sample_size_rng_parallel,
        )["data"]
        sample_size_run_parallel = su.plain_silverman_variant(
            sample_size_sample_parallel,
            cfg_parallel["k_modes_tested"],
            b_boot=min(cfg_parallel["b_boot"], 50),
            max_iter=40,
            tol=1e-6,
        )
        return sample_size_run_parallel["p_value"]

    n_values = np.array([60, 120, 240, 480, 960])
    mean_p = []

    for n_now in n_values:
        sample_size_rep_p_values = Parallel(n_jobs=-1, backend="loky")(
            delayed(run_sample_size_parallel)(n_now, rep_idx_n_parallel, scenario, cfg)
            for rep_idx_n_parallel in range(min(cfg["n_reps"], 12))
        )
        mean_p.append(np.mean(sample_size_rep_p_values))

    mean_p = np.asarray(mean_p)
    return mean_p, n_values


@app.cell
def _(hv, mean_p, n_values, save):
    pval_curve= hv.Curve((n_values, mean_p)).opts(
        color="#7c3aed",
        line_width=3,
        xlabel="sample size",
        ylabel="mean p-value",
        title="Average conclusion as sample size changes",
    )
    save(pval_curve, filename='close_bimodal_pvalue_samplesize.html')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
