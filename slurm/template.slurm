#!/bin/bash
# ============================================================================
# Standard SLURM template for thesis jobs on ALICE.
# ============================================================================

#SBATCH --job-name=thesis            
#SBATCH --partition=testing        
#SBATCH --gres=gpu:1                 
#SBATCH --time=00:30:00              
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%j.out      
#SBATCH --error=logs/%x_%j.out       

set -euo pipefail

# ---------------------------------------------------------------------------
# PER-JOB KNOBS 
# ---------------------------------------------------------------------------
ENV=/zfsstore/user/s4184343/thesis-env      
DATASET=$HOME/datasets/pets                 
CMD="python extract.py --data \$SCRATCH/data --out \$SCRATCH/out"
# ---------------------------------------------------------------------------

# --- environment ---
export HF_HUB_OFFLINE=1                       
source "$ENV/bin/activate"                    

# --- fast per-job scratch (WIPED at job end) ---
SCRATCH=/scratchdata/${SLURM_JOB_USER}/${SLURM_JOB_ID}
mkdir -p "$SCRATCH/data" "$SCRATCH/out"
cp -r "$DATASET"/. "$SCRATCH/data/"

echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- run ---
eval "$CMD"

# --- copy results OFF scratch before it disappears ---
mkdir -p "$HOME/thesis/results"
cp -r "$SCRATCH/out"/. "$HOME/thesis/results/"

echo "[$(date)] done."