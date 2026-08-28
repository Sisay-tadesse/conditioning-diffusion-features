#!/bin/bash

set -euo pipefail

for w in 7.5 10; do
    sbatch slurm/e1_extract.slurm pets 200 P2 "$w"
    sbatch slurm/e1_extract.slurm cub  300 P2 "$w"
done

squeue -u "$USER"
