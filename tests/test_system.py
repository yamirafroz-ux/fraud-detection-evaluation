import sqlite3
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fluxfraud.config import Config
from fluxfraud.evaluation import evaluate, metrics, split_masks
from fluxfraud.features import FEATURES, build_features
from fluxfraud.simulator import simulate
from fluxfraud.storage import export_run


@pytest.fixture(scope="module")
def world():
    c = Config(customers=30, days=10, compromise_hazard=0.006)
    return c, simulate(c)


def test_seed_and_ledger(world):
    c, s = world
    pd.testing.assert_frame_equal(s.transactions, simulate(c).transactions)
    assert not s.transactions.equals(simulate(replace(c, seed=43)).transactions)
    balances = s.accounts.initial_balance_cents.to_numpy().copy()
    for r in s.transactions.itertuples():
        assert balances[r.sender] == r.balance_before_cents
        assert r.sender != r.receiver
        assert r.amount_cents > 0
        assert r.accepted == (balances[r.sender] >= r.amount_cents)
        if r.accepted:
            balances[r.sender] -= r.amount_cents
            balances[r.receiver] += r.amount_cents
    assert np.array_equal(balances, s.accounts.final_balance_cents)
    assert balances.sum() == s.accounts.initial_balance_cents.sum()
    assert (balances >= 0).all()


def test_features_are_prefix_invariant_and_label_blind(world):
    _, s = world
    tx = s.transactions
    full = build_features(tx)
    pd.testing.assert_frame_equal(full.iloc[:100], build_features(tx.iloc[:100]))
    altered = tx.assign(is_fraud=1 - tx.is_fraud, typology="unknown", label_available_at=-1)
    pd.testing.assert_frame_equal(full, build_features(altered))
    assert full.iloc[0].count_1h == 0
    assert np.isfinite(full.to_numpy()).all()
    assert not set(FEATURES) & {"is_fraud", "typology", "accepted", "label_available_at", "sender"}
    with pytest.raises(ValueError):
        build_features(tx.iloc[::-1])


def test_split_label_maturity(world):
    c, s = world
    masks = split_masks(s.transactions, c)
    assert (s.transactions.loc[masks["train"], "label_available_at"] <= c.horizon * 0.5).all()
    assert (s.transactions.loc[masks["validation"], "label_available_at"] <= c.horizon * 0.7).all()
    assert not (masks["train"] & masks["test"]).any()


def test_daily_capacity_and_zero_positive_metrics():
    tx = pd.DataFrame(
        dict(time=[1, 2, 25, 26], is_fraud=[1, 0, 0, 1], amount_cents=[100] * 4, accepted=[1] * 4)
    )
    m = metrics(tx, [0.9, 0.8, 0.8, 0.9], 0.85, 1)
    assert m["precision_at_daily_budget"] == 1
    assert m["recall_at_daily_budget"] == 1
    assert m["settled_fraud_value_capture"] == 1
    assert metrics(tx.assign(is_fraud=0), [0] * 4, 1, 1)["average_precision"] is None


def test_end_to_end_and_sql_export(world, tmp_path):
    c, s = world
    features = build_features(s.transactions)
    result, pred, policies = evaluate(s.transactions, features, c)
    assert len(result) == 18
    assert set(pred.columns) >= {"boosting", "logistic", "isolation_forest"}
    export_run(tmp_path / "run", c, s, features, result, pred, policies)
    with sqlite3.connect(tmp_path / "run/transactions.sqlite") as con:
        assert con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == len(s.transactions)
    with pytest.raises(FileExistsError):
        export_run(tmp_path / "run", c, s, features, result, pred, policies)


def test_no_fraud_fails_clearly():
    c = Config(customers=10, days=10, compromise_hazard=0)
    s = simulate(c)
    with pytest.raises(ValueError, match="both classes"):
        evaluate(s.transactions, build_features(s.transactions), c)


def test_budget_shared_across_shift_boundary():
    from fluxfraud.evaluation import daily_selection

    tx = pd.DataFrame({"time": [25, 26, 27, 28]})
    selected = daily_selection(tx, [0.1, 0.8, 0.9, 0.2], 1)
    assert selected.sum() == 1
    assert selected[:2].sum() + selected[2:].sum() == 1


def test_hand_calculated_history():
    tx = pd.DataFrame(
        {
            "transaction_id": [0, 1, 2, 3],
            "time": [0.0, 0.5, 1.0, 24.0],
            "sender": [0] * 4,
            "receiver": [1, 1, 2, 1],
            "amount_cents": [100] * 4,
            "foreign": [0] * 4,
            "new_device": [0] * 4,
            "balance_before_cents": [1000] * 4,
        }
    )
    f = build_features(tx)
    assert list(f.count_1h) == [0, 1, 1, 0]
    assert list(f.count_24h) == [0, 1, 2, 2]
    assert list(f.prior_pair_count) == [0, 1, 0, 2]
    assert f.iloc[3].sender_entropy == pytest.approx(-2 / 3 * np.log(2 / 3) - 1 / 3 * np.log(1 / 3))
