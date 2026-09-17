"""Chronological, delayed-label evaluation with validation-only policy selection."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .features import FEATURES, STATIC, TEMPORAL


def split_masks(tx, config):
    train_end, test_start = config.horizon * 0.5, config.horizon * 0.7
    return {
        "train": (tx.time < train_end) & (tx.label_available_at <= train_end),
        "validation": (tx.time >= train_end)
        & (tx.time < test_start)
        & (tx.label_available_at <= test_start),
        "test": tx.time >= test_start,
    }


def daily_selection(tx, scores, budget):
    """Select once over the whole test set; period slices share the same daily budget."""
    selected = np.zeros(len(tx), dtype=bool)
    scores = np.asarray(scores)
    days = np.floor(tx.time.to_numpy() / 24).astype(int)
    for day in np.unique(days):
        ids = np.flatnonzero(days == day)
        order = np.argsort(-scores[ids], kind="stable")
        selected[ids[order[:budget]]] = True
    return selected


def metrics(tx, scores, threshold, budget, selected=None):
    if len(tx) == 0:
        return {"n": 0, "frauds": 0, "average_precision": None, "roc_auc": None}
    y = tx.is_fraud.to_numpy()
    scores = np.asarray(scores)
    positive = scores >= threshold
    tp = int(y[positive].sum())
    if selected is None:
        selected = daily_selection(tx, scores, budget)
    frauds = int(y.sum())
    value = tx.amount_cents.to_numpy() * tx.accepted.to_numpy() * y
    return {
        "n": len(tx),
        "frauds": frauds,
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, scores)) if frauds else None,
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "precision": tp / int(positive.sum()) if positive.any() else 0.0,
        "recall": tp / frauds if frauds else None,
        "alert_rate": float(positive.mean()),
        "precision_at_daily_budget": float(y[selected].mean()) if selected.any() else None,
        "recall_at_daily_budget": int(y[selected].sum()) / frauds if frauds else None,
        "settled_fraud_value_capture": float(value[selected].sum() / value.sum())
        if value.sum()
        else None,
    }


@threadpool_limits.wrap(limits=1)
def evaluate(tx, features, config):
    if not tx.index.equals(features.index) or len(tx) != len(features):
        raise ValueError("Transactions and features must have identical indexes")
    if not np.isfinite(features[FEATURES].to_numpy()).all():
        raise ValueError("Features must be finite")
    masks = split_masks(tx, config)
    for name, mask in masks.items():
        if mask.sum() < 10:
            raise ValueError(f"Too few {name} observations; increase customers, days or rate")
    train, valid, test = (masks[k] for k in ("train", "validation", "test"))
    if tx.loc[train, "is_fraud"].nunique() != 2:
        raise ValueError("Training needs both classes; increase run size or compromise_hazard")
    models = {
        "logistic": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500)), FEATURES),
        "boosting": (
            HistGradientBoostingClassifier(
                max_iter=100, max_leaf_nodes=15, l2_regularization=2, random_state=config.seed
            ),
            FEATURES,
        ),
        "boosting_static": (
            HistGradientBoostingClassifier(
                max_iter=100, max_leaf_nodes=15, l2_regularization=2, random_state=config.seed
            ),
            STATIC,
        ),
        "boosting_no_network": (
            HistGradientBoostingClassifier(
                max_iter=100, max_leaf_nodes=15, l2_regularization=2, random_state=config.seed
            ),
            STATIC + TEMPORAL,
        ),
        "isolation_forest": (
            IsolationForest(n_estimators=100, random_state=config.seed, n_jobs=1),
            FEATURES,
        ),
    }
    rows, predictions, policies = (
        [],
        tx.loc[test, ["transaction_id", "time", "is_fraud"]].copy(),
        {},
    )
    validation_days = max((config.horizon * 0.2 - config.label_delay_hours) / 24, 1 / 24)
    fraction = min(config.alert_budget * validation_days / valid.sum(), 1.0)
    for name, (model, columns) in models.items():
        x_train = features.loc[train, columns]
        if name == "isolation_forest":
            model.fit(x_train)  # Unlabelled training mixture, not oracle-clean normal data.
            val_scores = -model.score_samples(features.loc[valid, columns])
            scores = -model.score_samples(features.loc[test, columns])
        else:
            model.fit(x_train, tx.loc[train, "is_fraud"])
            val_scores = model.predict_proba(features.loc[valid, columns])[:, 1]
            scores = model.predict_proba(features.loc[test, columns])[:, 1]
        threshold = float(np.quantile(val_scores, 1 - fraction, method="higher"))
        policies[name] = {"threshold": threshold, "validation_target_alert_fraction": fraction}
        predictions[name] = scores
        test_tx = tx.loc[test]
        selected = daily_selection(test_tx, scores, config.alert_budget)
        for period, subset in {
            "all_test": np.ones(test.sum(), dtype=bool),
            "pre_shift": test_tx.time.to_numpy() < config.drift_fraction * config.horizon,
            "post_shift": test_tx.time.to_numpy() >= config.drift_fraction * config.horizon,
        }.items():
            result = metrics(
                test_tx.loc[subset],
                scores[subset],
                threshold,
                config.alert_budget,
                selected[subset],
            )
            if name != "isolation_forest" and subset.any():
                result["brier_score"] = float(
                    brier_score_loss(test_tx.loc[subset, "is_fraud"], scores[subset])
                )
            rows.append({"model": name, "period": period, **result})
    # Transparent operational baseline, expressed only in observable quantities.
    rule = (
        features.log_amount_z.to_numpy()
        + 2 * features.new_device.to_numpy()
        + features.foreign.to_numpy()
    )
    threshold = float(np.quantile(rule[valid], 1 - fraction, method="higher"))
    predictions["behaviour_rule"] = rule[test]
    policies["behaviour_rule"] = {
        "threshold": threshold,
        "validation_target_alert_fraction": fraction,
    }
    selected = daily_selection(tx.loc[test], rule[test], config.alert_budget)
    for period, subset in {
        "all_test": np.ones(test.sum(), dtype=bool),
        "pre_shift": tx.loc[test, "time"].to_numpy() < config.drift_fraction * config.horizon,
        "post_shift": tx.loc[test, "time"].to_numpy() >= config.drift_fraction * config.horizon,
    }.items():
        rows.append(
            {
                "model": "behaviour_rule",
                "period": period,
                **metrics(
                    tx.loc[test].loc[subset],
                    rule[test][subset],
                    threshold,
                    config.alert_budget,
                    selected[subset],
                ),
            }
        )
    return pd.DataFrame(rows), predictions, policies
