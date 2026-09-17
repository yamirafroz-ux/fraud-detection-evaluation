# Literature and design decisions

This is a focused technical reading of foundational work, not a systematic review or claim to cover every recent fraud method. Equations and mechanisms in the repository are an original synthesis. Reading a method in another domain does not establish its empirical validity for retail transfers.

## Hawkes dynamics

**Bacry, Mastromatteo & Muzy (2015), Hawkes processes in finance.**
[Full text](https://arxiv.org/html/1502.04592), especially Section 2, cluster interpretation, and Appendix B.

The review distinguishes endogenous excitation from exogenous intensity and explains both thinning and cluster simulation. We use diagonal exponential kernels and piecewise-constant backgrounds; we do not import empirical order-book conclusions into retail fraud. The homogeneous Poisson limit and long-run mean are regression checks. Near-critical excitation amplifies simulation cost, motivating an explicit resource guard. Seasonality and changing latent states mean full-run stationarity must not be assumed.

## Activity and temporal networks

**Perra, Gonçalves, Pastor-Satorras & Vespignani (2012), Activity driven modeling of time varying networks.**
[Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC3384079/), model definition, network aggregation and Discussion.

Their activity-driven construction separates individual activation from aggregate topology and demonstrates consequences of aggregation for spreading dynamics. Their basic model omits persistent memory. FluxFraud borrows the emphasis on event time, but deliberately adds reinforcement and Gibbs recipient preferences. Consequently its network is not a reproduction of their analytical model. Final-graph centralities are unavailable to an earlier fraud decision; causal graph features are computed incrementally.

## Delayed supervision and operational constraints

**Dal Pozzolo, Boracchi, Caelen, Alippi & Bontempi (2015), Credit Card Fraud Detection and Concept-Drift Adaptation with Delayed Supervised Information.**
[Author-hosted paper](https://boracchi.faculty.polimi.it/docs/2015_04_Credit_Card_Fraud_Detection_DalPozzolo_Boracchi_Caelen_Alippi_Bontempi.pdf), Sections III–IV.

The paper distinguishes recent investigator-selected feedback from delayed labels and discusses their different distributions. Our fixed-delay maturity filter addresses only one part of that problem. We deliberately freeze detectors rather than claim to reproduce its adaptive learning scheme. Daily top-K ranking is a batch abstraction: even the paper notes that an actual continuous system cannot always wait for the whole day's transactions. A credible extension needs explicit review queues and policy-dependent label selection.

## Synthetic financial environments

**Lopez-Rojas, Axelsson & Baca (2018), Analysis of fraud controls using the PaySim financial simulator.**
[Publisher abstract and bibliographic record](https://www.inderscience.com/info/e_inarticle.php?artid=93756), DOI: 10.1504/IJSPM.2018.093756.

The accessible abstract describes aggregate-data-driven synthetic transactions with malicious behaviour injected for control analysis. Only the abstract was accessible for this source; this repository does not claim a full-text replication. It motivates controlled experimentation and the central importance of calibration. FluxFraud has no equivalent empirical calibration data, so it makes weaker realism claims and reports its assumptions openly.

## Rare-event evaluation

**Saito & Rehmsmeier (2015), The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.**
[Open-access paper](https://doi.org/10.1371/journal.pone.0118432), Results and Discussion.

The study shows how precision–recall views reveal the positive-prediction burden under imbalance. It does not imply ROC AUC is mathematically invalid. We report both ranking summaries alongside prevalence and review capacity. Our primary summary is average precision, whose implementation differs from trapezoidal PR integration. Comparing AP across shifted periods mixes ranking changes and prevalence changes; both must be inspected.

## What is an assumption rather than a literature result?

The hourly behaviour matrix, hazard values, amount distributions, hidden ring size, energy weights, device probabilities and combined shift are modelling choices. They were not estimated from Revolut or any bank. OU relaxation and Gibbs selection give those assumptions a coherent mathematical structure, but do not validate their numerical values. The appropriate next step is sensitivity analysis and comparison to permitted aggregate observations.

## Falsifiable research questions

1. Holding model capacity fixed, do temporal features improve performance across independent worlds?
2. Does network memory help detection, or mainly create spurious confidence tied to this generator?
3. How much apparent robustness comes from changes in prevalence rather than discrimination?
4. Does a simple behavioural rule generalise better than a nonlinear classifier under camouflage?
5. How do the relaxation times 1/kappa and 1/beta compare with label latency and review periods?

The current multi-seed and ablation commands answer parts of questions 1–4. Question 5 needs explicit parameter sweeps. No result should be claimed before running its experiment.
