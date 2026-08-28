#!/bin/bash
#SBATCH --job-name=e2_smoke
#SBATCH --partition=testing
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node857
#SBATCH --output=logs/%x_%j.out

# All 3 baselines on 10 Pets images: do the primed caches load offline on a
# compute node? That is the failure --prime cannot catch.
# Writes to features_smoke/ so it cannot clobber anything.
set -euo pipefail

source slurm/env.sh

echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/extract_baselines.py --dataset pets --limit 10 \
    --out "${FEATURES}_smoke"

echo "=== E2 smoke PASSED ==="
echo "cleanup: rm -rf ${FEATURES}_smoke ; then: sbatch slurm/e2_baselines.slurm"
