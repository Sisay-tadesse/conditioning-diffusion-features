#!/bin/bash
#SBATCH --job-name=e5-lora
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --exclude=node857
#SBATCH --output=logs/%x_%j.out

#   sbatch slurm/lora_train.slurm pets
# -> ~/lora/<dataset>_r16/ckpt-{1000,2500,5000}/
set -euo pipefail

DATASET=$1

source slurm/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # headroom on 11GB

echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# dataset -> scratch
SCRATCH=/scratchdata/${SLURM_JOB_USER}/${SLURM_JOB_ID}
mkdir -p "$SCRATCH/data/$DATASET"
cp -r "$DATA/$DATASET"/. "$SCRATCH/data/$DATASET/"


python src/train_lora.py --dataset "$DATASET" --data "$SCRATCH/data" \
    --out "$LORA_DIR/${DATASET}_r16"

echo "=== lora done: $(ls $LORA_DIR/${DATASET}_r16) ==="
