"""Independent Monte Carlo worlds, mechanism interventions and seed-level intervals."""

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import t

from .evaluation import evaluate
from .features import build_features
from .simulator import simulate


def experiment(config, seeds, scenarios=("full", "no_excitation", "no_memory", "no_shift")):
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Provide distinct, nonempty seeds")
    interventions = {
        "full": {},
        "no_excitation": {"branching_ratio": 0.0},
        "no_memory": {"network_memory": 0.0},
        "no_shift": {"drift_strength": 0.0},
    }
    if not set(scenarios).issubset(interventions) or not scenarios:
        raise ValueError("Unknown or empty scenarios")
    results = []
    for scenario in scenarios:
        for seed in seeds:
            cfg = replace(config, seed=seed, **interventions[scenario])
            sim = simulate(cfg)
            result, _, _ = evaluate(sim.transactions, build_features(sim.transactions), cfg)
            results.append(result.assign(seed=seed, scenario=scenario))
    raw = pd.concat(results, ignore_index=True)
    summary = []
    for keys, group in raw.groupby(["scenario", "model", "period"]):
        for metric in ("average_precision", "precision_at_daily_budget", "recall_at_daily_budget"):
            values = group[metric].dropna().to_numpy()
            count = len(values)
            mean = float(values.mean()) if count else np.nan
            se = float(values.std(ddof=1) / np.sqrt(count)) if count > 1 else np.nan
            width = t.ppf(0.975, count - 1) * se if count > 1 else np.nan
            summary.append(
                dict(zip(("scenario", "model", "period"), keys))
                | {
                    "metric": metric,
                    "n_seeds": count,
                    "mean": mean,
                    "standard_error": se,
                    "ci95_low": mean - width,
                    "ci95_high": mean + width,
                }
            )
    return raw, pd.DataFrame(summary)
