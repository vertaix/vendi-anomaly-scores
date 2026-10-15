#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from public_benchmarks import (  # noqa: E402
    METHOD_REGISTRY,
    aggregate_trial_rows,
    build_trial_dataset,
    evaluate_method,
    load_embedding_bundle,
)


IMAGE_METHODS = ["KNN", "TorchGMM", "VAS"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark public methods on ImageNet embedding trials with fixed samples "
            "per selected class and a fixed overall anomaly rate."
        )
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "imagenet")
    parser.add_argument("--embedding-dirs", nargs="+", default=["dinov3_vitb16_embeddings"])
    parser.add_argument("--methods", nargs="+", choices=IMAGE_METHODS, default=list(IMAGE_METHODS))
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--n-normal-classes", type=int, default=25)
    parser.add_argument("--n-anomaly-classes", type=int, default=25)
    parser.add_argument("--normal-samples-per-class", type=int, default=1000)
    parser.add_argument("--anomaly-rate", type=float, default=0.2)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "imagenet_fixed_samples")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--knn-device", default="auto")
    parser.add_argument("--knn-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--gmm-min-components", type=int, default=1)
    parser.add_argument("--gmm-max-components", type=int, default=10)
    parser.add_argument("--gmm-reg-covar", type=float, default=1e-4)
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
    if args.n_normal_classes <= 0 or args.n_anomaly_classes <= 0:
        raise ValueError("Class counts must be positive.")
    if args.normal_samples_per_class <= 0:
        raise ValueError("--normal-samples-per-class must be positive.")
    if not 0.0 < args.anomaly_rate < 1.0:
        raise ValueError("--anomaly-rate must be in the range (0, 1).")
    return args


def get_eligible_wnids(bundle, min_samples_per_class):
    eligible_wnids = [
        wnid
        for wnid in bundle["wnids"].tolist()
        if int(bundle["class_to_indices"][wnid].size) >= min_samples_per_class
    ]
    return np.array(sorted(eligible_wnids), dtype=object)


def distribute_counts_evenly(total_count, n_classes, rng):
    base_count = total_count // n_classes
    remainder = total_count % n_classes
    counts = np.full(n_classes, base_count, dtype=np.int64)
    if remainder:
        extra_indices = rng.choice(n_classes, size=remainder, replace=False)
        counts[extra_indices] += 1
    return counts


def sample_trial_indices(bundle, rng, n_normal_classes, n_anomaly_classes, normal_samples_per_class, anomaly_rate):
    eligible_wnids = get_eligible_wnids(bundle, normal_samples_per_class)
    if n_normal_classes + n_anomaly_classes > int(eligible_wnids.size):
        raise ValueError(
            "Not enough eligible classes to satisfy the requested normal and anomaly counts."
        )

    normal_wnids = np.sort(rng.choice(eligible_wnids, size=n_normal_classes, replace=False))
    remaining_wnids = np.setdiff1d(eligible_wnids, normal_wnids, assume_unique=True)
    anomaly_wnids = np.sort(rng.choice(remaining_wnids, size=n_anomaly_classes, replace=False))

    normal_index_chunks = []
    for wnid in normal_wnids.tolist():
        class_indices = bundle["class_to_indices"][wnid]
        sampled_indices = rng.choice(class_indices, size=normal_samples_per_class, replace=False)
        normal_index_chunks.append(sampled_indices)
    normal_indices = np.concatenate(normal_index_chunks).astype(np.int64, copy=False)
    n_normals = int(normal_indices.size)

    target_n_anomalies = int(round(anomaly_rate * n_normals / (1.0 - anomaly_rate)))
    anomaly_counts_array = distribute_counts_evenly(target_n_anomalies, n_anomaly_classes, rng)
    anomaly_index_chunks = []
    for wnid, sample_count in zip(anomaly_wnids.tolist(), anomaly_counts_array.tolist()):
        class_indices = bundle["class_to_indices"][wnid]
        if sample_count > int(class_indices.size):
            raise ValueError(
                f"Anomaly class {wnid} does not have enough samples: "
                f"requested {sample_count}, available {int(class_indices.size)}."
            )
        if sample_count == 0:
            continue
        sampled_indices = rng.choice(class_indices, size=sample_count, replace=False)
        anomaly_index_chunks.append(sampled_indices)

    anomaly_indices = (
        np.concatenate(anomaly_index_chunks).astype(np.int64, copy=False)
        if anomaly_index_chunks
        else np.empty(0, dtype=np.int64)
    )
    return normal_wnids, anomaly_wnids, normal_indices, anomaly_indices


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundles = [load_embedding_bundle(args.data_root, name, args.split) for name in args.embedding_dirs]
    rows = []
    for bundle_idx, bundle in enumerate(bundles):
        for trial in range(args.n_trials):
            trial_seed = args.seed + 1000 * bundle_idx + trial
            rng = np.random.default_rng(trial_seed)
            normal_wnids, anomaly_wnids, normal_indices, anomaly_indices = sample_trial_indices(
                bundle=bundle,
                rng=rng,
                n_normal_classes=args.n_normal_classes,
                n_anomaly_classes=args.n_anomaly_classes,
                normal_samples_per_class=args.normal_samples_per_class,
                anomaly_rate=args.anomaly_rate,
            )
            _, X, y = build_trial_dataset(bundle, normal_indices, anomaly_indices)
            for method_name in args.methods:
                row = {
                    "dataset": bundle["embedding_name"],
                    "method": method_name,
                    "seed": int(trial_seed),
                    "trial": int(trial),
                    "status": "ok",
                    "message": "",
                    "roc_auc": float("nan"),
                    "ap": float("nan"),
                    "runtime_seconds": float("nan"),
                    "n_normals": int(normal_indices.size),
                    "n_anomalies": int(anomaly_indices.size),
                    "normal_wnids": "|".join(normal_wnids.tolist()),
                    "anomaly_wnids": "|".join(anomaly_wnids.tolist()),
                }
                try:
                    out = evaluate_method(method_name=method_name, X=X, y=y, seed=trial_seed, args=args)
                except Exception as exc:
                    row["status"] = "error"
                    row["message"] = f"{exc.__class__.__name__}: {exc}"
                    rows.append(row)
                    continue

                row["roc_auc"] = out["roc_auc"]
                row["ap"] = out["ap"]
                row["runtime_seconds"] = out["runtime_seconds"]
                rows.append(row)

    detailed_df = pd.DataFrame(rows)
    detailed_path = args.output_dir / "imagenet_fixed_samples_trial_results.csv"
    summary_path = args.output_dir / "imagenet_fixed_samples_trial_summary.csv"
    detailed_df.to_csv(detailed_path, index=False)
    aggregate_trial_rows(detailed_df).to_csv(summary_path, index=False)

    print(f"Wrote detailed results to {detailed_path}")
    print(f"Wrote summary results to {summary_path}")


if __name__ == "__main__":
    main()
