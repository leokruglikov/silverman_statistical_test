import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import holoviews as hv
    import numpy as np
    import silverman_utils as su
    from holoviews import opts
    from scipy.signal import find_peaks

    opts.defaults(
        opts.Curve(width=850, height=380),
        opts.Histogram(width=850, height=380),
        opts.Scatter(width=850, height=380, size=8),
    )
    return find_peaks, hv, mo, np, su


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
def _(h_small, hist, plot_kde, search, x, x_grid):
    (
        hist
        * plot_kde(x, x_grid, h_small, "#2563eb")
        * plot_kde(x, x_grid, search["h_crit"], "#dc2626")
    ).opts(title="Blue: less smoothing, Red: critical bandwidth")
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
    hs = np.linspace(max(search["h_crit"] * 0.15, 1e-3), search["h_crit"] * 2.2, 50)
    scan = su.summarize_mode_scan(scenario["data"], hs, x_grid=x_grid)
    hv.Curve((scan["hs"], scan["n_modes"])).opts(
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Repeated draws

    A single picture can be convincing for the wrong reason.
    Repeating the same scenario shows whether the conclusion is stable or fragile.
    """)
    return


@app.cell
def _(cfg, np, scenario, su):
    p_values = []
    h_crit_values = []

    for rep_idx in range(cfg["n_reps"]):
        repeated_rng = np.random.default_rng(cfg["seed"] + rep_idx)
        repeated_sample = su.make_scenario(
            scenario["name"],
            n_samples=cfg["n_samples"],
            rng=repeated_rng,
        )["data"]
        repeated_run = su.plain_silverman_variant(
            repeated_sample,
            cfg["k_modes_tested"],
            b_boot=cfg["b_boot"],
            max_iter=40,
            tol=1e-6,
        )
        p_values.append(repeated_run["p_value"])
        h_crit_values.append(repeated_run["h_crit"])

    p_values = np.asarray(p_values)
    h_crit_values = np.asarray(h_crit_values)
    return h_crit_values, p_values


@app.cell
def _(h_crit_values, hv, np, p_values):
    p_freq, p_edges = np.histogram(p_values, bins=10, density=True)
    h_freq, h_edges = np.histogram(h_crit_values, bins=10, density=True)

    p_plot = hv.Histogram((p_edges, p_freq)).opts(
        color="#93c5fd",
        line_color="#1d4ed8",
        title="Repeated p-values",
    )
    h_plot = hv.Histogram((h_edges, h_freq)).opts(
        color="#fca5a5",
        line_color="#b91c1c",
        title="Repeated h_crit values",
    )
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
def _(cfg, np, scenario, su):
    n_values = np.array([60, 120, 240, 480, 960])
    mean_p = []

    for n_now in n_values:
        sample_size_rep_p_values = []
        for rep_idx_n in range(min(cfg["n_reps"], 12)):
            sample_size_rng = np.random.default_rng(10_000 + 100 * n_now + rep_idx_n)
            sample_size_sample = su.make_scenario(
                scenario["name"],
                n_samples=int(n_now),
                rng=sample_size_rng,
            )["data"]
            sample_size_run = su.plain_silverman_variant(
                sample_size_sample,
                cfg["k_modes_tested"],
                b_boot=min(cfg["b_boot"], 50),
                max_iter=40,
                tol=1e-6,
            )
            sample_size_rep_p_values.append(sample_size_run["p_value"])
        mean_p.append(np.mean(sample_size_rep_p_values))

    mean_p = np.asarray(mean_p)
    return mean_p, n_values


@app.cell
def _(hv, mean_p, n_values):
    hv.Curve((n_values, mean_p)).opts(
        color="#7c3aed",
        line_width=3,
        xlabel="sample size",
        ylabel="mean p-value",
        title="Average conclusion as sample size changes",
    )
    return


if __name__ == "__main__":
    app.run()
