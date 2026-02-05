#!/bin/bash
#SBATCH --account=def-qltian
#SBATCH --job-name=pu-osls-tabpfn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --signal=TERM@120

set -euo pipefail

# Optional: switch to your repo location in /work or /scratch on cluster
REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"

mkdir -p logs artifacts/cluster_checkpoints

# Helps with long-running variable-shape CUDA allocations.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export PYTHONUNBUFFERED=1

# Match the exact environment used during package installation.
module purge
module load python/3.10.13
module load cuda/12.2
source ~/venvs/tabpfn/bin/activate

# Ensure project source is importable even without editable install.
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

NPROC="${SLURM_GPUS_ON_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"

python -c "import torch; print('torch', torch.__version__)"

python -m torch.distributed.run \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  scripts/train_cluster.py \
  --num-steps 30000 \
  --global-batch-size 64 \
  --lr 4e-4 \
  --use-curriculum \
  --save-every-steps 500 \
  --checkpoint-dir artifacts/cluster_checkpoints \
  --resume-from artifacts/cluster_checkpoints/latest.pt
