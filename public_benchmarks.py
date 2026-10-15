from __future__ import annotations

import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler


METHOD_REGISTRY = {
    "ADERH": ("methods.aderh", "ADERH"),
    "IForest": ("methods.iforest", "IForest"),
    "KNN": ("methods.knn", "KNN"),
    "LODA": ("methods.loda", "LODA"),
    "LOF": ("methods.lof", "LOF"),
    "SLAD": ("methods.slad", "SLAD"),
    "TorchGMM": ("methods.torch_gmm", "TorchGMM"),
    "VAS": ("methods.vas", "VAS"),
}

STOCHASTIC_METHODS = {"ADERH", "IForest", "LODA", "SLAD", "TorchGMM"}
DEFAULT_SEEDS = [0, 1, 2, 1000, 10000]


def resolve_runtime_device(device_name: str) -> str:
    if device_name != "auto":
        return device_name
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_npz_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "X" not in data or "y" not in data:
        raise ValueError(f"{path} must contain 'X' and 'y' arrays.")
    X = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.int64).ravel()
    if X.ndim != 2:
        raise ValueError(f"{path} contains X with shape {X.shape}; expected 2D features.")
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"{path} contains mismatched X/y lengths: {X.shape[0]} vs {y.shape[0]}.")
    return X, y


def resolve_dataset_paths(data_dir: Path, dataset_filters: list[str] | None) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    data_paths = sorted(data_dir.glob("*.npz"))
    if not data_paths:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    if not dataset_filters:
        return data_paths

    wanted = set(dataset_filters)
    filtered = []
    for path in data_paths:
        if path.name in wanted or path.stem in wanted:
            filtered.append(path)
    if not filtered:
        raise FileNotFoundError(f"No datasets from {sorted(wanted)} were found in {data_dir}")
    return filtered


def maybe_preprocess(method_name: str, X: np.ndarray) -> np.ndarray:
    if method_name == "ADERH":
        return MinMaxScaler().fit_transform(X)
    return X


def score_model(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).ravel()
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X)).ravel()
    raise AttributeError(f"{model.__class__.__name__} does not expose decision_function/predict.")


def resolve_method_class(method_name: str):
    module_name, class_name = METHOD_REGISTRY[method_name]
    module = import_module(module_name)
    return getattr(module, class_name)


def build_model(method_name: str, seed: int, args):
    if method_name == "ADERH":
        return resolve_method_class(method_name)(
            n_estimators=args.aderh_n_estimators,
            n=args.aderh_n,
            random_state=seed,
        )
    if method_name == "IForest":
        return resolve_method_class(method_name)(
            random_state=seed,
            n_estimators=args.iforest_n_estimators,
            max_samples=args.iforest_max_samples,
        )
    if method_name == "KNN":
        return resolve_method_class(method_name)(
            n_neighbors=args.knn_k,
            device=resolve_runtime_device(args.knn_device),
            dtype=args.knn_dtype,
        )
    if method_name == "LODA":
        return resolve_method_class(method_name)(
            random_state=seed,
            n_random_cuts=args.loda_n_random_cuts,
            n_bins=args.loda_n_bins,
        )
    if method_name == "LOF":
        return resolve_method_class(method_name)(n_neighbors=args.lof_n_neighbors)
    if method_name == "SLAD":
        return resolve_method_class(method_name)(
            random_state=seed,
            device=resolve_runtime_device(args.slad_device),
            verbose=0,
        )
    if method_name == "TorchGMM":
        return resolve_method_class(method_name)(
            min_components=args.gmm_min_components,
            max_components=args.gmm_max_components,
            reg_covar=args.gmm_reg_covar,
            n_init=args.gmm_n_init,
            max_iter=args.gmm_max_iter,
            random_state=seed,
            device=resolve_runtime_device(args.gmm_device),
            tol=args.gmm_tol,
            init_iters=args.gmm_init_iters,
            dtype=args.gmm_dtype,
        )
    if method_name == "VAS":
        return resolve_method_class(method_name)(
            q_mode=args.vas_q_mode,
            fixed_q=args.vas_fixed_q,
            adaptive_q_floor=args.vas_adaptive_q_floor,
            rank=args.vas_rank,
            device=resolve_runtime_device(args.vas_device),
            eps=args.vas_eps,
        )
    raise ValueError(f"Unsupported method: {method_name}")


def evaluate_method(method_name: str, X: np.ndarray, y: np.ndarray, seed: int, args) -> dict:
    model = build_model(method_name=method_name, seed=seed, args=args)
    X_eval = maybe_preprocess(method_name, X)
    started = time.perf_counter()
    model.fit(X_eval)
    scores = score_model(model, X_eval)
    runtime_seconds = float(time.perf_counter() - started)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite anomaly scores produced.")
    return {
        "scores": scores,
        "roc_auc": float(roc_auc_score(y, scores)),
        "ap": float(average_precision_score(y, scores)),
        "runtime_seconds": runtime_seconds,
    }


def aggregate_trial_rows(detailed_df: pd.DataFrame) -> pd.DataFrame:
    ok_df = detailed_df[detailed_df["status"] == "ok"].copy()
    if ok_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "method",
                "successful_trials",
                "attempted_trials",
                "roc_auc_mean",
                "roc_auc_std",
                "ap_mean",
                "ap_std",
                "runtime_seconds_mean",
                "runtime_seconds_std",
            ]
        )

    grouped = (
        ok_df.groupby(["dataset", "method"], as_index=False)
        .agg(
            successful_trials=("seed", "count"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            ap_mean=("ap", "mean"),
            ap_std=("ap", "std"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
            runtime_seconds_std=("runtime_seconds", "std"),
        )
    )
    attempted = (
        detailed_df.groupby(["dataset", "method"], as_index=False)
        .agg(attempted_trials=("seed", "count"))
    )
    summary = grouped.merge(attempted, on=["dataset", "method"], how="left")
    for col in ["roc_auc_std", "ap_std", "runtime_seconds_std"]:
        summary[col] = summary[col].fillna(0.0)
    return summary.sort_values(["dataset", "method"]).reset_index(drop=True)


def load_embedding_bundle(data_root: Path, embedding_dir_name: str, split: str) -> dict:
    bundle_dir = data_root / embedding_dir_name
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Embedding bundle not found: {bundle_dir}")
    metadata_paths = sorted(bundle_dir.glob("*_metadata.csv"))
    embedding_paths = sorted(bundle_dir.glob("*_embeddings.npy"))
    if not metadata_paths or not embedding_paths:
        raise FileNotFoundError(
            f"{bundle_dir} must contain *_metadata.csv and *_embeddings.npy files."
        )

    metadata = pd.read_csv(metadata_paths[0])
    split_mask = metadata["split"].to_numpy() == split
    split_metadata = metadata.loc[split_mask].reset_index(drop=True)
    if split_metadata.empty:
        raise ValueError(f"No rows with split={split!r} were found in {metadata_paths[0]}.")
    embeddings = np.load(embedding_paths[0], mmap_mode="r")

    class_to_indices = {}
    for wnid, group in split_metadata.groupby("wnid", sort=True):
        class_to_indices[wnid] = group["index"].to_numpy(dtype=np.int64)

    return {
        "embedding_name": embedding_dir_name,
        "bundle_dir": bundle_dir,
        "metadata_path": metadata_paths[0],
        "embeddings_path": embedding_paths[0],
        "embeddings": embeddings,
        "split_metadata": split_metadata,
        "class_to_indices": class_to_indices,
        "wnids": np.array(sorted(class_to_indices), dtype=object),
    }


def build_trial_dataset(bundle: dict, normal_indices: np.ndarray, anomaly_indices: np.ndarray):
    selected_indices = np.concatenate([normal_indices, anomaly_indices]).astype(np.int64, copy=False)
    labels = np.concatenate(
        [
            np.zeros(normal_indices.shape[0], dtype=np.int64),
            np.ones(anomaly_indices.shape[0], dtype=np.int64),
        ]
    )
    features = np.asarray(bundle["embeddings"][selected_indices], dtype=np.float64)
    return selected_indices, features, labels
