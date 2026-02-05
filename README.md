# PU/OSLS TabPFN (synthetic prior)

Research codebase for training and evaluating a lightweight TabPFN-style transformer on synthetic tabular tasks for:
- positive-unlabeled (PU) learning
- open-set label shift (OSLS)

## Project layout

```text
.
├── pyproject.toml
├── requirement.txt
├── scripts/
│   ├── train_cluster.py
│   └── slurm/
│       ├── train_pretraining_multigpu.sbatch
│       └── train_pretraining_smoketest.sbatch
├── src/
│   └── pu_osls_tabpfn/
│       ├── __init__.py
│       ├── prior_data.py
│       ├── prior_data_legacy.py
│       ├── tabicl_prior/
│       │   └── *.py
│       ├── model.py
│       ├── eval_pu_osls.py
│       └── train.py
├── tests/
│   └── test_prior_data.py
└── notebooks/
    └── pretraining.ipynb
```

## Installation

```bash
pip install -e .
```

Install pinned dependencies:

```bash
pip install -r requirement.txt
```

For development tools:

```bash
pip install -e ".[dev]"
```

Optional (for TabICL `tree_scm` / `mix_scm` priors):

```bash
pip install -e ".[tree-prior]"
```

## Usage

Train with defaults:

```bash
pu-osls-train
```

or:

```bash
python -m pu_osls_tabpfn.train
```

Tune run settings:

```bash
pu-osls-train --num-steps 1000 --batch-size 8 --eval-interval 100 --eval-tasks 200 --seed 0 --prior-backend tabicl
```

Enable curriculum learning (simple -> complex datasets):

```bash
pu-osls-train --use-curriculum --curriculum-update-every-steps 500 --curriculum-max-updates 20 --curriculum-start-max-classes 2 --curriculum-start-max-features 5
```

Use legacy prior generator:

```bash
pu-osls-train --prior-backend legacy
```

Notebook workflow:

```bash
jupyter notebook notebooks/pretraining.ipynb
```

## Cluster training

Run distributed training with checkpoint resume:

```bash
torchrun --nproc_per_node=4 scripts/train_cluster.py \
  --use-curriculum \
  --global-batch-size 64 \
  --num-steps 30000 \
  --save-every-steps 200 \
  --checkpoint-dir artifacts/cluster_checkpoints \
  --resume-from artifacts/cluster_checkpoints/latest.pt
```

Submit Slurm jobs:

```bash
sbatch scripts/slurm/train_pretraining_multigpu.sbatch
sbatch scripts/slurm/train_pretraining_smoketest.sbatch
```

## Method summary

For each synthetic task:
1. Generate full datasets with TabICL prior (`src/pu_osls_tabpfn/tabicl_prior`).
2. Keep all classes in both train and test for the full dataset stage.
3. Apply PU/OSLS construction by removing sampled classes from **train rows only** (Poisson-based count).
4. Train with a reserved unseen class (`max_classes + 1` output head).
5. Evaluate outlier detection + seen-class + overall mapped accuracy.

Notes:
- `prior_data.py` is the default TabICL-backed data module.
- `prior_data_legacy.py` keeps the previous linear synthetic generator.
- Test-set label shift hooks are scaffolded but currently a no-op (future extension).
- In curriculum mode, `min_features` stays fixed; only `max_features` expands over update stages.

## References

- Grinsztajn et al., *TabPFN* (ICML 2023)
- Aigul et al., *TabICL* (NeurIPS 2023)
