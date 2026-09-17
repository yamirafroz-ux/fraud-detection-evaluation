"""Command-line entry points with explicit output ownership."""

import argparse
from pathlib import Path

from .config import Config
from .evaluation import evaluate
from .experiments import experiment
from .features import build_features
from .simulator import simulate
from .storage import export_run


def main():
    parser = argparse.ArgumentParser(description="FluxFraud stochastic experiment laboratory")
    parser.add_argument("command", choices=["run", "monte-carlo"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists; choose a fresh directory to preserve earlier results")
    try:
        config = Config.load(args.config) if args.config else Config()
        if args.command == "run":
            simulation = simulate(config)
            features = build_features(simulation.transactions)
            results, predictions, policies = evaluate(simulation.transactions, features, config)
            export_run(args.out, config, simulation, features, results, predictions, policies)
            print(results.loc[results.period == "all_test"].to_string(index=False))
        else:
            raw, summary = experiment(config, args.seeds)
            args.out.mkdir(parents=True)
            config.save(args.out / "config.json")
            raw.to_csv(args.out / "replicates.csv", index=False)
            summary.to_csv(args.out / "summary.csv", index=False)
        print(f"Saved results to {args.out.resolve()}")
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
