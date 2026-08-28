#!/bin/bash

set -euo pipefail

for w in 0 1 2 3 5 7.5; do
    sbatch slurm/e1_extract.slurm pets 200 P3 "$w"
    sbatch slurm/e1_extract.slurm cub  300 P3 "$w"
done

# deployable arm, smaller grid:
for w in 2 5; do
    sbatch slurm/e1_extract.slurm pets 200 P2 "$w"
    sbatch slurm/e1_extract.slurm cub  300 P2 "$w"
done

squeue -u "$USER"