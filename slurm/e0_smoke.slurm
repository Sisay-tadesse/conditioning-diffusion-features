#!/bin/bash
#SBATCH --job-name=e0-smoke
#SBATCH --partition=testing
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node857
#SBATCH --output=logs/%x_%j.out

# Extractor end to end on 10 Pets images. testing partition = short queue.
# mkdir -p logs && sbatch slurm/e0_smoke.slurm
set -euo pipefail

source slurm/env.sh

echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/extract_features.py --dataset pets --split trainval --t 100 --limit 10 \
    --out "$FEATURES"

echo "=== E0 smoke PASSED ==="
