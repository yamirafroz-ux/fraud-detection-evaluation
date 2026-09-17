# Measured experiment report

Configuration: `configs/default.json`; reference seed 42; Monte Carlo seeds 11, 22, 33, 44, 55.
Twenty simulated worlds (five seeds × four mechanism settings), with six detector variants per world. No seed was selected or discarded based on performance.

[Open the interactive results chart](experiment_chart.html) · [All replicates](monte_carlo_replicates.csv) · [Summary and uncertainty](monte_carlo_summary.csv)

## Full simulator: average precision

| Detector | Pre-shift mean | Post-shift mean | All-test mean |
|---|---:|---:|---:|
| behaviour_rule | 0.593 | 0.085 | 0.149 |
| boosting | 0.525 | 0.075 | 0.134 |
| boosting_no_network | 0.584 | 0.083 | 0.144 |
| boosting_static | 0.498 | 0.088 | 0.133 |
| isolation_forest | 0.284 | 0.045 | 0.061 |
| logistic | 0.627 | 0.089 | 0.161 |

Uncertainty intervals in the linked CSV are Student-t intervals across five independent seeds. The short pre-shift test slice contains relatively few frauds; uncertainty is substantial.

## Interpretation

All six detector variants lose average precision after the intervention. This intervention changes multiple distributions together, including prevalence, so the difference is not an isolated causal estimate of camouflage.

Logistic regression has the highest mean all-test AP in this five-seed pilot. Full boosting does not show a reliable advantage from network features. The hand-written behaviour rule remains competitive. This is evidence against assuming that more complex features automatically generalise better in this environment.

## Paired feature comparisons

Differences below compare variants on each identical realised world; intervals are computed on the seed-level differences, not by subtracting two marginal intervals.

| Comparison (all-test AP) | Mean difference | 95% interval |
|---|---:|---:|
| boosting − boosting_static | +0.0007 | [-0.0668, +0.0681] |
| boosting − boosting_no_network | -0.0103 | [-0.0410, +0.0204] |
| boosting − behaviour_rule | -0.0155 | [-0.0710, +0.0400] |

These are exploratory comparisons with no multiplicity correction. Five seeds are insufficient for confident model selection. Hyperparameters were fixed before these runs; no test-based tuning is represented.

## Scope

The reference manifest records runtime versions and the transaction digest. These results were produced locally; the GitHub CI workflow has not yet been run remotely. The data generator is uncalibrated and the scores do not estimate bank performance. Reproduce the study with the command in README.md, preserving the exact environment for bit-level comparisons.
