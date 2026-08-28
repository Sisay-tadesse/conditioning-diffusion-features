#!/bin/bash
#SBATCH --job-name=e1-extract
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --exclude=node857
#SBATCH --output=logs/%x_%j.out

# One grid cell: all 5 blocks at one t, both splits (the probe needs a pair).
#
#   sbatch slurm/e1_extract.slurm <dataset> <t> [prompt] [cfg] [lora_path]
#   for t in 50 100 200 300 500; do sbatch slurm/e1_extract.slurm pets $t; done
#   sbatch slurm/e1_extract.slurm pets 100 P3 3.0
#   sbatch slurm/e1_extract.slurm pets 100 P3 "" $LORA_DIR/pets_r16
set -euo pipefail

DATASET=$1
T=$2
PROMPT=${3:-P0}
CFG=${4:-}
LORA=${5:-}

source slurm/env.sh

echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES"
echo "dataset=$DATASET t=$T prompt=$PROMPT cfg=${CFG:-off} lora=${LORA:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# dataset -> scratch, wiped at job end
SCRATCH=/scratchdata/${SLURM_JOB_USER}/${SLURM_JOB_ID}
mkdir -p "$SCRATCH/data/$DATASET" "$SCRATCH/out"
cp -r "$DATA/$DATASET"/. "$SCRATCH/data/$DATASET/"

EXTRA=()
[[ -n "$CFG" ]] && EXTRA+=(--cfg "$CFG")
[[ -n "$LORA" ]] && EXTRA+=(--lora "$LORA")

if [[ "$DATASET" == "pets" ]]; then SPLITS="trainval test"; else SPLITS="train test"; fi
for SPLIT in $SPLITS; do
    python src/extract_features.py --dataset "$DATASET" --split "$SPLIT" \
        --t "$T" --prompt "$PROMPT" --data "$SCRATCH/data" \
        --out "$SCRATCH/out" "${EXTRA[@]}"
done

# get the outputs off scratch before it goes
mkdir -p "$FEATURES"
cp -r "$SCRATCH/out"/. "$FEATURES/"
echo "=== done: $(ls "$SCRATCH/out") ==="