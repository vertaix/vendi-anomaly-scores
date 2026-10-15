#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/adbench}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/adbench}"

EMBEDDING_DIRS=(
  "CV_by_ResNet18"
  "CV_by_ViT"
  "NLP_by_BERT"
  "NLP_by_RoBERTa"
)

METHODS=(
  "VAS"
  "TorchGMM"
  "KNN"
  "ADERH"
  "LOF"
  "LODA"
  "IForest"
  "SLAD"
)

mkdir -p "${RESULTS_ROOT}"

for embedding_dir in "${EMBEDDING_DIRS[@]}"; do
  data_dir="${DATA_ROOT}/${embedding_dir}"
  for method in "${METHODS[@]}"; do
    output_dir="${RESULTS_ROOT}/${embedding_dir}/${method}"
    mkdir -p "${output_dir}"
    python "${REPO_ROOT}/scripts/run_adbench_embeddings.py" \
      --method "${method}" \
      --data-dir "${data_dir}" \
      --results-dir "${output_dir}" \
      --knn-device auto \
      --slad-device auto \
      --gmm-device auto \
      --vas-device auto
  done
done
