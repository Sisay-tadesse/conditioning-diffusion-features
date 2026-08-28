#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --partition=testing
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%j.out


set -euo pipefail

source slurm/env.sh

python - <<'PY'
import os, torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    os.environ["SD15"], torch_dtype=torch.float16, safety_checker=None).to("cuda")

# poke the UNet directly, it is what the hooks hang off
lat = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16)
t = torch.tensor([100], device="cuda")
ids = pipe.tokenizer("", return_tensors="pt").input_ids.to("cuda")
emb = pipe.text_encoder(ids)[0].half()
out = pipe.unet(lat, t, encoder_hidden_states=emb).sample
print("UNet OK, output:", tuple(out.shape))
PY

echo "[$(date)] done."
