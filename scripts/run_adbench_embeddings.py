#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from public_benchmarks import (  # noqa: E402
    DEFAULT_SEEDS,
    METHOD_REGISTRY,
    STOCHASTIC_METHODS,
    aggregate_trial_rows,
    evaluate_method,
    load_npz_dataset,
    resolve_dataset_paths,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one public method over an ADBENCH embedding directory.",
    )
    parser.add_argument("--method", required=True, choices=sorted(METHOD_REGISTRY))
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory containing .npz datasets.")
    parser.add_argument("--results-dir", required=True, type=Path, help="Directory where outputs are written.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Optional dataset stems or filenames.")
    parser.add_argument("--summary-file", default="summary.csv")
    parser.add_argument(
        "--score-tag",
        default="",
        help="Optional suffix appended to score filenames before the .csv extension.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--knn-device", default="auto")
    parser.add_argument("--knn-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--aderh-n-estimators", type=int, default=256)
    parser.add_argument("--aderh-n", type=int, default=18)
    parser.add_argument("--iforest-n-estimators", type=int, default=100)
    parser.add_argument("--iforest-max-samples", type=int, default=256)
    parser.add_argument("--lof-n-neighbors", type=int, default=20)
    parser.add_argument("--loda-n-random-cuts", type=int, default=100)
    parser.add_argument("--loda-n-bins", type=int, default=10)
    parser.add_argument("--slad-device", default="auto")
    parser.add_argument("--gmm-min-components", type=int, default=1)
    parser.add_argument("--gmm-max-components", type=int, default=10)
    parser.add_argument("--gmm-reg-covar", type=float, default=1e-6)
    parser.add_argument("--gmm-n-init", type=int, default=1)
    parser.add_argument("--gmm-max-iter", type=int, default=100)
    parser.add_argument("--gmm-device", default="auto")
    parser.add_argument("--gmm-tol", type=float, default=1e-3)
    parser.add_argument("--gmm-init-iters", type=int, default=10)
    parser.add_argument("--gmm-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--vas-q-mode", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--vas-fixed-q", type=float, default=None)
    parser.add_argument("--vas-adaptive-q-floor", type=float, default=0.1)
    parser.add_argument("--vas-rank", default="full")
    parser.add_argument("--vas-device", default="auto")
    parser.add_argument("--vas-eps", type=float, default=1e-12)
    args = parser.parse_args()

    if args.vas_q_mode == "fixed" and args.vas_fixed_q is None:
        raise ValueError("--vas-fixed-q is required when --vas-q-mode=fixed.")
    if args.vas_q_mode == "adaptive" and args.vas_fixed_q is not None:
        raise ValueError("--vas-fixed-q cannot be used with --vas-q-mode=adaptive.")
    return args


def resolve_seeds(method_name: str, user_seeds: list[int] | None) -> list[int]:
    if user_seeds:
        return [int(seed) for seed in user_seeds]
    if method_name in STOCHASTIC_METHODS:
        return list(DEFAULT_SEEDS)
    return [0]


def score_filename(dataset_path: Path, method_name: str, seed: int, deterministic: bool, score_tag: str) -> str:
    stem = f"{dataset_path.stem}__{method_name}"
    if not deterministic:
        stem += f"__seed_{seed}"
    if score_tag:
        stem += f"__{score_tag}"
    return f"{stem}.csv"


def write_score_csv(path: Path, y, scores):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "label", "score"])
        for idx, (label, score) in enumerate(zip(y.tolist(), scores.tolist())):
            writer.writerow([idx, int(label), float(score)])


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    score_dir = args.results_dir / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths = resolve_dataset_paths(args.data_dir, args.datasets)
    seeds = resolve_seeds(args.method, args.seeds)
    deterministic = len(seeds) == 1 and args.method not in STOCHASTIC_METHODS

    rows = []
    for dataset_path in dataset_paths:
        X, y = load_npz_dataset(dataset_path)
        for seed in seeds:
            row = {
                "dataset": dataset_path.stem,
                "method": args.method,
                "seed": int(seed),
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "status": "ok",
                "message": "",
                "roc_auc": float("nan"),
                "ap": float("nan"),
                "runtime_seconds": float("nan"),
                "score_file": "",
            }
            try:
                out = evaluate_method(method_name=args.method, X=X, y=y, seed=seed, args=args)
            except Exception as exc:
                row["status"] = "error"
                row["message"] = f"{exc.__class__.__name__}: {exc}"
                rows.append(row)
                continue

            score_path = score_dir / score_filename(
                dataset_path=dataset_path,
                method_name=args.method,
                seed=seed,
                deterministic=deterministic,
                score_tag=args.score_tag,
            )
            write_score_csv(score_path, y=y, scores=out["scores"])

            row["roc_auc"] = out["roc_auc"]
            row["ap"] = out["ap"]
            row["runtime_seconds"] = out["runtime_seconds"]
            row["score_file"] = str(score_path.relative_to(args.results_dir))
            rows.append(row)

    detailed_df = pd.DataFrame(rows)
    detailed_path = args.results_dir / "trial_results.csv"
    summary_path = args.results_dir / args.summary_file
    detailed_df.to_csv(detailed_path, index=False)
    aggregate_trial_rows(detailed_df).to_csv(summary_path, index=False)

    print(f"Wrote detailed results to {detailed_path}")
    print(f"Wrote summary results to {summary_path}")


if __name__ == "__main__":
    main()
