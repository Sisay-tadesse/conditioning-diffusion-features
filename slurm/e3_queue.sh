#!/bin/bash
# Prompt ladder, 16 jobs. t values are the grid optima plus their neighbour.
# P0 already exists from the grid run.
# bash slurm/e3_queue.sh
set -euo pipefail

for t in 100 200; do
    for p in P1 P2 P3 P4; do
        sbatch slurm/e1_extract.slurm pets "$t" "$p"
    done
done

for t in 200 300; do
    for p in P1 P2 P3 P4; do
        sbatch slurm/e1_extract.slurm cub "$t" "$p"
    done
done

squeue -u "$USER"
