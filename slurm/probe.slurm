#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node857
#SBATCH --output=logs/%x_%j.out

# Probe every un-probed pair in ~/features. Idempotent, so rerun it whenever
# new features land.
set -euo pipefail

source slurm/env.sh

echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID"

python src/probe.py --dir "$FEATURES" --results results/results.csv

echo "=== probe done; commit results.csv ==="
