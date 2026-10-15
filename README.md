# Vendi Anomaly Scores

Repository for *Vendi Anomaly Scores for Efficient and Accurate Anomaly Detection*, to appear at NeurIPS 2026. 

This repository contains code to reproduce the benchmark results reported in our paper, including our ADBENCH and ImageNet experiments. 

## Install

Create a Python environment and install the benchmark dependencies:

```bash
pip install -r requirements.txt
```

It is enough for the public ADBENCH benchmark and the fixed-samples ImageNet evaluation once embeddings already exist.

The optional ImageNet embedding helper script needs extra packages that are not part of the core benchmark requirements:
- `transformers`
- `datasets`
- `huggingface-hub`

## Methods

Methods in this repo:
- `VAS`
- `TorchGMM` (GMD in the paper)
- `KNN` (DN2)
- `ADERH`
- `LOF`
- `LODA`
- `IForest`
- `SLAD`

## ADBENCH Data Setup

This repo expects precomputed ADBENCH embedding datasets stored as `.npz` files with `X` and `y` arrays.

The upstream ADBENCH repository provides the source datasets:
- `https://github.com/Minqi824/ADBench/tree/main/adbench/datasets`

For this repo, place the prepared embedding datasets under:

```text
vendi-anomaly-scores/
  data/
    adbench/
      CV_by_ResNet18/
        CIFAR10_0.npz
        ...
      CV_by_ViT/
        CIFAR10_0.npz
        ...
      NLP_by_BERT/
        imdb.npz
        amazon.npz
        ...
      NLP_by_RoBERTa/
        imdb.npz
        amazon.npz
        ...
```

### Verify the layout

Before running benchmarks, make sure each folder contains `.npz` files:

```bash
find data/adbench -maxdepth 2 -type f | grep '\.npz$'
```

You should see files under all four embedding directories.

## Run ADBENCH Benchmarks

Run one method on one embedding family:

```bash
python scripts/run_adbench_embeddings.py \
  --method VAS \
  --data-dir data/adbench/CV_by_ViT \
  --results-dir results/adbench/CV_by_ViT/VAS
```

Run the full public comparison suite:

```bash
bash scripts/sweep_adbench.sh
```

The ADBENCH runner writes:
- `trial_results.csv`
- `summary.csv`
- per-dataset score files under `scores/`

## ImageNet Preparation

For the ImageNet experiment, obtain the ImageNet-1K training images from the official ImageNet website.

Expected local layout:

```text
vendi-anomaly-scores/
  data/
    imagenet/
      imagenet_1k/
        train/
          n01440764/
            *.JPEG
        val/
          n01440764/
            *.JPEG
```

Embeddings for the reported experiment were produced with:
- model: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- representation: CLS-token embeddings
- normalization: row-wise L2 normalization

A sanitized reference script is included at:
- `scripts/embed_imagenet_dinov3.py`

Example usage:

```bash
python scripts/embed_imagenet_dinov3.py \
  --data-root data/imagenet/imagenet_1k \
  --split both \
  --out-dir data/imagenet/dinov3_vitb16_embeddings \
  --batch-size 128 \
  --num-workers 8
```

The fixed-samples ImageNet benchmark expects an embedding directory containing:
- `*_embeddings.npy`
- `*_metadata.csv`

## Run the ImageNet Fixed-Samples Benchmark

Run one setting manually:

```bash
python scripts/run_imagenet_fixed_samples.py \
  --data-root data/imagenet \
  --embedding-dirs dinov3_vitb16_embeddings \
  --methods VAS TorchGMM KNN \
  --split train \
  --n-trials 5 \
  --n-normal-classes 25 \
  --n-anomaly-classes 25 \
  --normal-samples-per-class 1000 \
  --anomaly-rate 0.2 \
  --output-dir results/imagenet_fixed_samples/example_run
```

Run the small sweep script:

```bash
bash scripts/sweep_imagenet_fixed_samples.sh
```

## Smoke Tests

Run the lightweight smoke test suite:

```bash
python scripts/smoke_test_public_repo.py
```

This script checks:
- imports for all public methods
- tiny synthetic ADBENCH runs for all methods
- clear failure on missing ADBENCH data paths
- tiny fake ImageNet embedding runs for `VAS`, `TorchGMM`, and `KNN`

## Citation

Coming soon!
