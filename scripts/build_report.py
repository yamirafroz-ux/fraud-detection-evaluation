"""Rebuild the portable measured-results report from checked-in CSVs."""

from pathlib import Path

import pandas as pd
import plotly.express as px
from scipy.stats import t

root = Path(__file__).resolve().parents[1]
reports = root / "reports"
summary = pd.read_csv(reports / "monte_carlo_summary.csv")
raw = pd.read_csv(reports / "monte_carlo_replicates.csv")
rows = summary[(summary.scenario == "full") & (summary.metric == "average_precision")]
fig = px.bar(
    rows[rows.period != "all_test"],
    x="model",
    y="mean",
    color="period",
    barmode="group",
    error_y=rows[rows.period != "all_test"].ci95_high - rows[rows.period != "all_test"]["mean"],
    template="plotly_dark",
    color_discrete_sequence=["#ff9f80", "#8be2d0"],
    title="Five independent worlds: ranking degrades after the joint shift",
    labels={"mean": "Mean average precision", "model": "Detector", "period": "Test period"},
)
fig.update_layout(height=620, margin=dict(b=160))
fig.write_html(reports / "experiment_chart.html", include_plotlyjs=True)
lines = [
    "# Measured experiment report",
    "",
    "Configuration: `configs/default.json`; reference seed 42; Monte Carlo seeds 11, 22, 33, 44, 55.",
    "Twenty simulated worlds (five seeds × four mechanism settings), with six detector variants per world. No seed was selected or discarded based on performance.",
    "",
    "[Open the interactive results chart](experiment_chart.html) · [All replicates](monte_carlo_replicates.csv) · [Summary and uncertainty](monte_carlo_summary.csv)",
    "",
    "## Full simulator: average precision",
    "",
    "| Detector | Pre-shift mean | Post-shift mean | All-test mean |",
    "|---|---:|---:|---:|",
]
for name, group in rows.groupby("model"):
    v = group.set_index("period")["mean"]
    lines.append(f"| {name} | {v['pre_shift']:.3f} | {v['post_shift']:.3f} | {v['all_test']:.3f} |")
lines.extend(
    [
        "",
        "Uncertainty intervals in the linked CSV are Student-t intervals across five independent seeds. The short pre-shift test slice contains relatively few frauds; uncertainty is substantial.",
        "",
        "## Interpretation",
        "",
        "All six detector variants lose average precision after the intervention. This intervention changes multiple distributions together, including prevalence, so the difference is not an isolated causal estimate of camouflage.",
        "",
        "Logistic regression has the highest mean all-test AP in this five-seed pilot. Full boosting does not show a reliable advantage from network features. The hand-written behaviour rule remains competitive. This is evidence against assuming that more complex features automatically generalise better in this environment.",
        "",
        "## Paired feature comparisons",
        "",
        "Differences below compare variants on each identical realised world; intervals are computed on the seed-level differences, not by subtracting two marginal intervals.",
        "",
        "| Comparison (all-test AP) | Mean difference | 95% interval |",
        "|---|---:|---:|",
    ]
)

full = raw[(raw.scenario == "full") & (raw.period == "all_test")].pivot(
    index="seed", columns="model", values="average_precision"
)
for other in ["boosting_static", "boosting_no_network", "behaviour_rule"]:
    delta = full.boosting - full[other]
    mean = delta.mean()
    width = t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / (len(delta) ** 0.5)
    lines.append(
        f"| boosting − {other} | {mean:+.4f} | [{mean - width:+.4f}, {mean + width:+.4f}] |"
    )
lines.extend(
    [
        "",
        "These are exploratory comparisons with no multiplicity correction. Five seeds are insufficient for confident model selection. Hyperparameters were fixed before these runs; no test-based tuning is represented.",
        "",
        "## Scope",
        "",
        "The reference manifest records runtime versions and the transaction digest. These results were produced locally; the GitHub CI workflow has not yet been run remotely. The data generator is uncalibrated and the scores do not estimate bank performance. Reproduce the study with the command in README.md, preserving the exact environment for bit-level comparisons.",
    ]
)
(reports / "RESULTS.md").write_text("\n".join(lines) + "\n")
