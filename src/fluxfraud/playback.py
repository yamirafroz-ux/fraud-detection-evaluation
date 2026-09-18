"""Build a deterministic visual replay of real simulator output, including causal cycles."""

import json
from collections import defaultdict, deque
from importlib.resources import files

MODEL_NAMES = {
    "logistic": "Logistic regression",
    "boosting": "Gradient boosting",
    "boosting_static": "Boosting · transaction only",
    "boosting_no_network": "Boosting · no network",
    "isolation_forest": "Isolation forest",
    "behaviour_rule": "Behaviour rule",
}


def temporal_loops(tx, window=24):
    """One shortest settled 2/3-edge temporal cycle per closing event, within 24 hours.

    Edges must occur in traversal order; no label or future graph is consulted.
    Multiple eligible cycles may exist; return one representative, not a census.
    """
    pairs = defaultdict(deque)
    expiry = deque()
    found = {}
    for r in tx.itertuples(index=False):
        t, i, j = r.time, r.sender, r.receiver
        while expiry and expiry[0][0] < t - window:
            old_t, a, b = expiry.popleft()
            q = pairs[a, b]
            q.popleft()
            if not q:
                del pairs[a, b]
        if not r.accepted:
            continue
        reverse = pairs.get((j, i))
        if reverse:
            found[int(r.transaction_id)] = [j, i, j]
        else:
            for (a, b), first in pairs.items():
                if a != j or b == i:
                    continue
                last = pairs.get((b, i))
                if last and first[0] < last[-1]:
                    found[int(r.transaction_id)] = [j, b, i, j]
                    break
        pairs[i, j].append(t)
        expiry.append((t, i, j))
    return found


def replay_html(config, sim, predictions, policies):
    tx = sim.transactions
    pred = predictions.set_index("transaction_id")
    names = list(MODEL_NAMES)
    flags = {
        int(idx): [int(row[n] >= policies[n]["threshold"]) for n in names]
        for idx, row in pred.iterrows()
    }
    loops = temporal_loops(tx)
    events = [
        [
            r.time,
            int(r.sender),
            int(r.receiver),
            int(r.amount_cents),
            int(r.is_fraud),
            int(r.accepted),
            flags.get(int(r.transaction_id)),
            loops.get(int(r.transaction_id)),
        ]
        for r in tx.itertuples(index=False)
    ]
    payload = {
        "events": events,
        "customers": config.customers,
        "days": config.days,
        "models": list(MODEL_NAMES.values()),
        "trainEnd": config.horizon * 0.5,
        "testStart": config.horizon * 0.7,
        "shift": config.horizon * config.drift_fraction,
        "delay": config.label_delay_hours,
        "budget": config.alert_budget,
    }
    template = files("fluxfraud").joinpath("ui/replay.html").read_text()
    return template.replace(
        "__PAYLOAD__", json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    )
