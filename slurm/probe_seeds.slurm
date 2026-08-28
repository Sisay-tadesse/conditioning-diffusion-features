#!/bin/bash
#SBATCH --job-name=probe-seeds
#SBATCH --partition=testing
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node857
#SBATCH --output=logs/%x_%j.out

# Error bars: seeds 1 and 2, seed 0 is already in the csv. Same cached
# features, so only the probe init and shuffling change.
# Idempotent per (file, block, seed) if the 30-min limit cuts it short.
set -euo pipefail

source slurm/env.sh

echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID"

for SEED in 1 2; do
    python src/probe.py --dir "$FEATURES" --results results/results.csv --seed "$SEED"
done

echo "=== seed reruns done; commit results.csv ==="
