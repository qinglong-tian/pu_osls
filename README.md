# PU/OSLS TabPFN

This repository contains the starting implementation for pretraining and evaluating a lightweight TabPFN-style transformer on synthetic tabular tasks for:
- positive-unlabeled (PU) learning
- open-set label shift (OSLS)

## What is implemented

- On-the-fly synthetic task generation with two backends:
  - `tabicl` (default): SCM-based priors in `src/pu_osls_tabpfn/tabicl_prior`
  - `legacy`: simple linear synthetic generator
- PU/OSLS task construction by removing selected classes from train rows only.
- Transformer classifier with a custom target encoder that uses a learnable unknown-label embedding for test rows.
- Training with optional curriculum learning (from easier to harder priors).
- Evaluation metrics for outlier detection and classification quality:
  - AUROC, AUPRC, TPR@FPR targets
  - seen-class accuracy / balanced accuracy
  - overall mapped accuracy / balanced accuracy
- Single-process training (`pu-osls-train`) and distributed training (`scripts/train_cluster.py`) with checkpoint resume.

## Installation

Editable install:

```bash
pip install -e .
```

Pinned runtime dependencies:

```bash
pip install -r requirement.txt
```

Optional extras:

```bash
pip install -e ".[dev]"         # pytest, ruff
pip install -e ".[tree-prior]"  # xgboost for tree_scm / mix_scm
```

## Quick start

Run default training:

```bash
pu-osls-train
```

Equivalent module call:

```bash
python -m pu_osls_tabpfn.train
```

Example with custom settings:

```bash
pu-osls-train \
  --num-steps 1000 \
  --batch-size 8 \
  --eval-interval 100 \
  --eval-tasks 200 \
  --seed 0 \
  --prior-backend tabicl
```

Enable curriculum:

```bash
pu-osls-train \
  --use-curriculum \
  --curriculum-update-every-steps 500 \
  --curriculum-max-updates 20 \
  --curriculum-start-max-classes 2 \
  --curriculum-start-max-features 5
```

Switch to the legacy generator:

```bash
pu-osls-train --prior-backend legacy
```

See all CLI options:

```bash
pu-osls-train --help
```

## Notebook workflow

Open:

```bash
jupyter notebook notebooks/pretraining.ipynb
```

The notebook covers:
- config setup
- batch sanity check
- optional GPU memory probe
- long-run curriculum training
- final evaluation and checkpoint save to `artifacts/pretrained_pu_osls_tabpfn.pt`

## Distributed / cluster training

Run multi-GPU training locally or on a node:

```bash
torchrun --nproc_per_node=4 scripts/train_cluster.py \
  --use-curriculum \
  --global-batch-size 64 \
  --num-steps 30000 \
  --save-every-steps 500 \
  --checkpoint-dir artifacts/cluster_checkpoints \
  --resume-from artifacts/cluster_checkpoints/latest.pt
```

Slurm examples:

```bash
sbatch scripts/slurm/train_pretraining_multigpu.sbatch
sbatch scripts/slurm/train_pretraining_smoketest.sbatch
```

## Tests

Run current test suite:

```bash
pytest
```

Current tests validate batch shapes and mask consistency for both `tabicl` and `legacy` backends.

## Repository structure

```text
.
├── src/pu_osls_tabpfn/
│   ├── model.py              # custom TabPFN-style model
│   ├── prior_data.py         # main PU/OSLS batch generator
│   ├── prior_data_legacy.py  # legacy generator
│   ├── eval_pu_osls.py       # evaluation metrics
│   ├── train.py              # single-process training
│   └── tabicl_prior/         # SCM priors, hp sampling, dataset generation utilities
├── scripts/train_cluster.py  # DDP training + checkpointing
├── scripts/slurm/            # Slurm job templates
├── notebooks/pretraining.ipynb
├── tests/test_prior_data.py
└── requirement.txt
```

## Current limitations

- Test-set label shift is scaffolded but not implemented (`TestLabelShiftConfig` beyond `"none"` raises `NotImplementedError`).
- `tree_scm` / `mix_scm` paths require `xgboost` and are significantly slower than `mlp_scm`.
