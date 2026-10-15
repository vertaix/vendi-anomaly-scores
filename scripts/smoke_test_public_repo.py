#!/usr/bin/env python3
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd, expect_success=True):
    out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if expect_success and out.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{out.stdout}\nSTDERR:\n{out.stderr}")
    if not expect_success and out.returncode == 0:
        raise RuntimeError(f"Command unexpectedly succeeded: {' '.join(cmd)}")
    return out


def make_adbench_dataset(path: Path, seed: int):
    rng = np.random.default_rng(seed)
    normal = rng.normal(loc=0.0, scale=1.0, size=(24, 8))
    anomaly = rng.normal(loc=3.0, scale=1.0, size=(8, 8))
    X = np.vstack([normal, anomaly]).astype(np.float32)
    y = np.concatenate([np.zeros(normal.shape[0]), np.ones(anomaly.shape[0])]).astype(np.int64)
    np.savez(path, X=X, y=y)


def make_imagenet_bundle(root: Path):
    bundle_dir = root / "toy_dinov3_embeddings"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    wnids = ["n0001", "n0002", "n0003", "n0004"]
    rows = []
    features = []
    idx = 0
    for wnid_idx, wnid in enumerate(wnids):
        center = wnid_idx * 2.0
        for image_idx in range(6):
            feat = rng.normal(loc=center, scale=0.25, size=(8,))
            features.append(feat.astype(np.float32))
            rows.append([idx, "train", f"/fake/{wnid}/{image_idx}.JPEG", f"{wnid}_{image_idx}", wnid, wnid])
            idx += 1

    np.save(bundle_dir / "facebook_dinov3-vitb16-pretrain-lvd1689m_embeddings.npy", np.stack(features))
    with (bundle_dir / "facebook_dinov3-vitb16-pretrain-lvd1689m_metadata.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "split", "path", "image_id", "wnid", "label_name"])
        writer.writerows(rows)


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        adbench_root = tmp / "adbench"
        adbench_root.mkdir()
        make_adbench_dataset(adbench_root / "toy_a.npz", seed=0)
        make_adbench_dataset(adbench_root / "toy_b.npz", seed=1)

        methods = ["VAS", "TorchGMM", "KNN", "ADERH", "LOF", "LODA", "IForest", "SLAD"]
        for method in methods:
            run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run_adbench_embeddings.py"),
                    "--method",
                    method,
                    "--data-dir",
                    str(adbench_root),
                    "--results-dir",
                    str(tmp / "results" / method),
                    "--aderh-n-estimators",
                    "8",
                    "--aderh-n",
                    "6",
                    "--iforest-n-estimators",
                    "20",
                    "--iforest-max-samples",
                    "16",
                    "--lof-n-neighbors",
                    "5",
                    "--loda-n-random-cuts",
                    "10",
                    "--loda-n-bins",
                    "5",
                    "--gmm-max-components",
                    "2",
                    "--gmm-max-iter",
                    "5",
                    "--gmm-n-init",
                    "1",
                    "--gmm-device",
                    "cpu",
                    "--knn-device",
                    "cpu",
                    "--slad-device",
                    "cpu",
                    "--vas-device",
                    "cpu",
                ]
            )

        run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_adbench_embeddings.py"),
                "--method",
                "VAS",
                "--data-dir",
                str(tmp / "missing"),
                "--results-dir",
                str(tmp / "bad"),
            ],
            expect_success=False,
        )

        imagenet_root = tmp / "imagenet"
        make_imagenet_bundle(imagenet_root)
        for method in ["VAS", "TorchGMM", "KNN"]:
            run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run_imagenet_fixed_samples.py"),
                    "--data-root",
                    str(imagenet_root),
                    "--embedding-dirs",
                    "toy_dinov3_embeddings",
                    "--methods",
                    method,
                    "--n-trials",
                    "1",
                    "--n-normal-classes",
                    "2",
                    "--n-anomaly-classes",
                    "2",
                    "--normal-samples-per-class",
                    "3",
                    "--anomaly-rate",
                    "0.25",
                    "--output-dir",
                    str(tmp / "imagenet_results" / method),
                    "--gmm-max-components",
                    "2",
                    "--gmm-max-iter",
                    "5",
                    "--gmm-device",
                    "cpu",
                    "--knn-device",
                    "cpu",
                    "--vas-device",
                    "cpu",
                ]
            )

    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
