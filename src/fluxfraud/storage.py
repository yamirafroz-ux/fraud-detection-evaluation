"""Portable SQLite export and reproducibility metadata."""

import hashlib
import importlib.metadata
import json
import platform
import sqlite3
from dataclasses import asdict
from pathlib import Path


def export_run(out, config, simulation, features, metrics, predictions, policies):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    config.save(out / "config.json")
    with sqlite3.connect(out / "transactions.sqlite") as con:
        for name, frame in {
            "transactions": simulation.transactions,
            "accounts": simulation.accounts,
            "latent_states": simulation.states,
            "features": features.assign(transaction_id=simulation.transactions.transaction_id),
            "predictions": predictions,
        }.items():
            frame.to_sql(name, con, index=False)
        con.execute("CREATE INDEX sender_time ON transactions(sender, time)")
        con.execute("CREATE INDEX receiver_time ON transactions(receiver, time)")
    metrics.to_csv(out / "metrics.csv", index=False)
    (out / "policies.json").write_text(json.dumps(policies, indent=2, allow_nan=False) + "\n")
    digest = hashlib.sha256(simulation.transactions.to_csv(index=False).encode()).hexdigest()
    manifest = {
        "config": asdict(config),
        "python": platform.python_version(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "pandas", "scikit-learn", "scipy", "fluxfraud")
        },
        "transaction_sha256": digest,
        "transactions": len(simulation.transactions),
        "warning": "Synthetic, uncalibrated experiment; not real-world performance.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
