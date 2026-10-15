#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/imagenet}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/imagenet_fixed_samples}"

NORMAL_CLASS_COUNTS=(25)
ANOMALY_CLASS_COUNTS=(25)
NORMAL_SAMPLES_PER_CLASS=1000
ANOMALY_RATES=(0.10 0.20 0.30)

METHODS=(
  "VAS"
  "TorchGMM"
  "KNN"
)

mkdir -p "${RESULTS_ROOT}"

for idx in "${!ANOMALY_RATES[@]}"; do
  anomaly_rate="${ANOMALY_RATES[$idx]}"
  anomaly_rate_tag="${anomaly_rate/./p}"
  output_dir="${RESULTS_ROOT}/normal_classes_${NORMAL_CLASS_COUNTS[0]}/anomaly_classes_${ANOMALY_CLASS_COUNTS[0]}/normal_samples_${NORMAL_SAMPLES_PER_CLASS}/anomaly_rate_${anomaly_rate_tag}"
  mkdir -p "${output_dir}"
  python "${REPO_ROOT}/scripts/run_imagenet_fixed_samples.py" \
    --data-root "${DATA_ROOT}" \
    --embedding-dirs "dinov3_vitb16_embeddings" \
    --methods "${METHODS[@]}" \
    --split train \
    --n-trials 5 \
    --seed 0 \
    --n-normal-classes "${NORMAL_CLASS_COUNTS[0]}" \
    --n-anomaly-classes "${ANOMALY_CLASS_COUNTS[0]}" \
    --normal-samples-per-class "${NORMAL_SAMPLES_PER_CLASS}" \
    --anomaly-rate "${anomaly_rate}" \
    --output-dir "${output_dir}" \
    --knn-device auto \
    --gmm-device auto \
    --vas-device auto
done
