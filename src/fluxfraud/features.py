"""Predictable features: read history BEFORE inserting the current event.

No ground-truth labels, future graph, latent attributes, acceptance decisions or
label-availability fields are allowed in the feature matrix.
"""

from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd

STATIC = ["log_amount", "hour_sin", "hour_cos", "foreign", "new_device", "balance_ratio"]
TEMPORAL = ["count_1h", "count_24h", "log_amount_z", "seconds_since_last"]
NETWORK = ["new_counterparty", "prior_pair_count", "receiver_senders", "sender_entropy"]
FEATURES = STATIC + TEMPORAL + NETWORK


def build_features(transactions):
    required = {
        "transaction_id",
        "time",
        "sender",
        "receiver",
        "amount_cents",
        "foreign",
        "new_device",
        "balance_before_cents",
    }
    if not required.issubset(transactions.columns):
        raise ValueError(f"Missing fields: {sorted(required - set(transactions.columns))}")
    if transactions.transaction_id.duplicated().any():
        raise ValueError("Transaction IDs must be unique")
    if not transactions.time.is_monotonic_increasing or transactions.time.duplicated().any():
        raise ValueError("Events must have strictly increasing timestamps")
    numeric = transactions[list(required)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (transactions.amount_cents <= 0).any():
        raise ValueError("Finite fields and positive amounts required")
    windows = defaultdict(deque)
    short = defaultdict(deque)
    moments = defaultdict(lambda: [0, 0.0, 0.0])
    last = {}
    pairs = defaultdict(Counter)
    incoming = defaultdict(set)
    sum_c_log_c = defaultdict(float)
    records = []
    for r in transactions.itertuples(index=False):
        i, j, t = r.sender, r.receiver, r.time
        for q, window in ((windows[i], 24), (short[i], 1)):
            while q and q[0] <= t - window:
                q.popleft()
        log_amount = np.log1p(r.amount_cents / 100)
        count, mean, m2 = moments[i]
        z = (log_amount - mean) / max(np.sqrt(m2 / max(count - 1, 1)), 0.3) if count > 1 else 0
        entropy = np.log(count) - sum_c_log_c[i] / count if count else 0
        records.append(
            [
                log_amount,
                np.sin(2 * np.pi * t / 24),
                np.cos(2 * np.pi * t / 24),
                r.foreign,
                r.new_device,
                r.amount_cents / max(r.balance_before_cents, 1),
                len(short[i]),
                len(windows[i]),
                z,
                min((t - last.get(i, t - 24)) * 3600, 86400),
                int(j not in pairs[i]),
                pairs[i][j],
                len(incoming[j]),
                entropy,
            ]
        )
        previous = pairs[i][j]
        sum_c_log_c[i] += (previous + 1) * np.log(previous + 1)
        if previous:
            sum_c_log_c[i] -= previous * np.log(previous)
        pairs[i][j] += 1
        incoming[j].add(i)
        windows[i].append(t)
        short[i].append(t)
        last[i] = t
        delta = log_amount - mean
        mean += delta / (count + 1)
        moments[i] = [count + 1, mean, m2 + delta * (log_amount - mean)]
    return pd.DataFrame(records, columns=FEATURES, index=transactions.index)
