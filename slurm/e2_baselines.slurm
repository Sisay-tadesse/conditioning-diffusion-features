#!/bin/bash
#SBATCH --job-name=e2_baselines
#SBATCH --partition=testing
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node857
#SBATCH --output=logs/%x_%j.out

# DINOv2 / CLIP / ResNet-50, all datasets and splits.
# Prime the caches on the login node first, or this fails offline:
#   python src/extract_baselines.py --prime
# Then rerun probe.slurm for the new rows.
set -euo pipefail

source slurm/env.sh

echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID"

python src/extract_baselines.py --out "$FEATURES"

echo "=== E2 extraction done; now: sbatch slurm/probe.slurm ==="
