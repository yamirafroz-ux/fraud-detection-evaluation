"""A marked transaction process with latent customer dynamics and conserved money."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .processes import gibbs_probabilities, hawkes_interval, ou_step


@dataclass
class Simulation:
    transactions: pd.DataFrame
    accounts: pd.DataFrame
    states: pd.DataFrame


def simulate(config: Config) -> Simulation:
    # Separate streams make latent dynamics insensitive to the number of event draws.
    latent_rng, event_rng, mark_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence(config.seed).spawn(3)
    ]
    n = config.customers
    group = latent_rng.integers(0, 4, n)
    log_mean = latent_rng.normal(np.log(45), 0.65, n)
    activity = latent_rng.lognormal(-0.5 * 0.55**2, 0.55, n)
    ring = latent_rng.choice(n, max(3, n // 20), replace=False)
    ring_set = set(ring)
    # Integer minor units avoid rounding drift in the closed ledger.
    initial = latent_rng.integers(200_000, 800_000, n, dtype=np.int64)
    balances = initial.copy()
    x = log_mean.copy()
    behaviour = np.zeros(n, dtype=int)
    compromised = np.zeros(n, dtype=bool)
    excitation = np.zeros(n)
    edge_counts = np.zeros((n, n), dtype=int)
    rows, states = [], []
    transition = np.array([[0.990, 0.006, 0.004], [0.035, 0.960, 0.005], [0.080, 0.010, 0.910]])
    for hour in range(config.horizon):
        shift = config.drift_strength if hour >= config.drift_fraction * config.horizon else 0
        behaviour = (latent_rng.random(n)[:, None] > np.cumsum(transition[behaviour], axis=1)).sum(
            axis=1
        )
        recover = latent_rng.random(n) < -np.expm1(-config.recovery_hazard)
        infect = latent_rng.random(n) < -np.expm1(-config.compromise_hazard * (1 + shift))
        compromised = np.where(compromised, ~recover, infect)
        target = log_mean + 0.20 * shift + np.choose(behaviour, [0.0, 0.25, 0.65])
        x = ou_step(x, target, config.ou_kappa, config.ou_sigma, 1, latent_rng)
        daily = 1 + 0.55 * np.cos(2 * np.pi * ((hour % 24) - 15) / 24)
        base = config.base_rate * activity * daily * np.choose(behaviour, [1, 1.4, 1.8])
        base *= np.where(compromised, 3.0 / (1 + 0.5 * shift), 1.0)
        pending = []
        for i in range(n):
            offsets, excitation[i] = hawkes_interval(
                base[i],
                excitation[i],
                config.branching_ratio,
                config.decay,
                1,
                event_rng,
                max_events=config.max_events - len(rows) - len(pending),
            )
            pending.extend((hour + t, i) for t in offsets)
        for t, i in sorted(pending):
            if len(rows) >= config.max_events:
                raise RuntimeError("Global event safety limit reached; reduce run size")
            fraud = bool(compromised[i] and mark_rng.random() < 0.65)
            candidates = np.delete(np.arange(n), i)
            energy = 1.4 * (group[candidates] != group[i]).astype(float)
            energy -= config.network_memory * np.log1p(edge_counts[i, candidates])
            if fraud:
                # Concentrated destinations create a recoverable network signature.
                energy -= (3.5 / (1 + shift)) * np.isin(candidates, ring)
            p = gibbs_probabilities(energy, config.temperature)
            j = int(mark_rng.choice(candidates, p=p))
            multiplier = (2.8 / (1 + 1.4 * shift)) if fraud else 1.0
            amount = max(1, int(round(100 * np.exp(mark_rng.normal(x[i], 0.60)) * multiplier)))
            foreign = bool(
                mark_rng.random()
                < (
                    0.45
                    if behaviour[i] == 1
                    else (0.55 / (1 + shift) if fraud else 0.04 + 0.03 * min(shift, 2))
                )
            )
            new_device = bool(mark_rng.random() < (0.65 / (1 + shift) if fraud else 0.035))
            before = int(balances[i])
            accepted = before >= amount
            if accepted:
                balances[i] -= amount
                balances[j] += amount
            edge_counts[i, j] += 1
            rows.append(
                (
                    len(rows),
                    t,
                    i,
                    j,
                    amount,
                    before,
                    int(accepted),
                    int(foreign),
                    int(new_device),
                    int(fraud),
                    "mule_transfer"
                    if fraud and i in ring_set
                    else "account_takeover"
                    if fraud
                    else "legitimate",
                    t + config.label_delay_hours,
                )
            )
        states.append(
            (
                hour,
                int(compromised.sum()),
                float(x.mean()),
                float(base.sum()),
                float(excitation.sum()),
                float(shift),
            )
        )
    tx = pd.DataFrame(
        rows,
        columns=[
            "transaction_id",
            "time",
            "sender",
            "receiver",
            "amount_cents",
            "balance_before_cents",
            "accepted",
            "foreign",
            "new_device",
            "is_fraud",
            "typology",
            "label_available_at",
        ],
    )
    accounts = pd.DataFrame(
        {
            "account_id": np.arange(n),
            "community": group,
            "initial_balance_cents": initial,
            "final_balance_cents": balances,
            "latent_activity": activity,
            "latent_ring": np.isin(np.arange(n), ring),
        }
    )
    return Simulation(
        tx,
        accounts,
        pd.DataFrame(
            states,
            columns=[
                "hour",
                "compromised",
                "mean_log_amount",
                "base_intensity",
                "excitation",
                "shift",
            ],
        ),
    )
