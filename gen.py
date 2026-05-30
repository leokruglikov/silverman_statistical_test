def simulate_random_dataset(kind, n=200, seed=None):
    rng = np.random.default_rng(seed)

    if kind == "random_unimodal":
        mu = rng.normal(0, 1)
        sigma = rng.uniform(0.5, 2.0)
        skew = rng.uniform(0.0, 1.0)

        if rng.uniform() < 0.5:
            x = rng.normal(mu, sigma, n)
        else:
            x = rng.lognormal(mean=mu, sigma=skew + 0.2, size=n)

    elif kind == "random_bimodal":
        weight = rng.uniform(0.25, 0.75)
        sep = rng.uniform(1.0, 5.0)
        s1 = rng.uniform(0.4, 1.4)
        s2 = rng.uniform(0.4, 1.4)

        z = rng.uniform(size=n) < weight
        x = np.empty(n)
        x[z] = rng.normal(-sep / 2, s1, z.sum())
        x[~z] = rng.normal(sep / 2, s2, (~z).sum())

    elif kind == "random_shoulder":
        weight = rng.uniform(0.75, 0.95)
        sep = rng.uniform(1.0, 2.5)
        s1 = rng.uniform(0.7, 1.5)
        s2 = rng.uniform(0.2, 0.8)

        z = rng.uniform(size=n) < weight
        x = np.empty(n)
        x[z] = rng.normal(0, s1, z.sum())
        x[~z] = rng.normal(sep, s2, (~z).sum())

    elif kind == "random_contaminated":
        eps = rng.uniform(0.01, 0.10)
        outlier_loc = rng.uniform(3.0, 8.0)
        outlier_scale = rng.uniform(0.1, 0.8)

        z = rng.uniform(size=n) < 1 - eps
        x = np.empty(n)
        x[z] = rng.normal(0, 1, z.sum())
        x[~z] = rng.normal(outlier_loc, outlier_scale, (~z).sum())

    else:
        raise ValueError(f"Unknown kind: {kind}")

    return x
