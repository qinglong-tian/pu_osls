#!/bin/bash
#SBATCH --job-name=pu-osls-smoke
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --signal=TERM@60

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"

mkdir -p logs artifacts/cluster_checkpoints

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export PYTHONUNBUFFERED=1

module load miniconda3/24.3.0-py311
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tabpfn-hpc

export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

NPROC=1
MASTER_PORT="${MASTER_PORT:-29510}"

python -m torch.distributed.run \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  scripts/train_cluster.py \
  --num-steps 200 \
  --global-batch-size 16 \
  --lr 2e-4 \
  --use-curriculum \
  --save-every-steps 50 \
  --checkpoint-dir artifacts/cluster_checkpoints_smoke \
  --resume-from artifacts/cluster_checkpoints_smoke/latest.pt
