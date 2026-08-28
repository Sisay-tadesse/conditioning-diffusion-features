# Conditioning Diffusion Features for Visual Recognition

MSc thesis code, LIACS. Can the conditioning pathway of a pretrained diffusion
model — prompt, classifier-free guidance, LoRA — improve the features it computes
for recognition?

Mostly no. Deployable prompts move linear-probe accuracy by under a point. A
rank-16 LoRA adapter at 5000 steps moves it by under two, with inconsistent sign.
Guidance on a domain-level prompt is the one lever that pays: +4.4 points on
CUB-200. Prompts naming the class do far better than any of these, but they leak
the label through cross-attention rather than describing the image; the sweeps
showing that are in here too.

Nothing samples an image. Noise a real image to timestep t, one UNet forward,
pool the block activations, probe them. The predicted noise is discarded.

## Layout

```
src/       extraction, baselines, probing, LoRA training
slurm/     the ALICE jobs these were run as
scripts/   the figures, built from results.csv
results/   results.csv — one row per (config, block, seed)
lora/      trained adapters at 1000 / 2500 / 5000 steps
```

Not here: feature caches (`*.pt`, regenerable), the datasets, and the SD-1.5
weights.

## Datasets

Neither dataset is redistributed here — both come with their own terms. Download
them yourself and arrange them like this under `$DATA`:

```
$DATA/
  pets/                       https://www.robots.ox.ac.uk/~vgg/data/pets/
    images/                     images.tar.gz
    annotations/                annotations.tar.gz
      trainval.txt
      test.txt
  cub/                        https://data.caltech.edu/records/65de6-vp158
    CUB_200_2011/               CUB_200_2011.tgz, extracted as-is
      images/
      images.txt
      image_class_labels.txt
      train_test_split.txt
      classes.txt
```

An extra `pets/oxford-iiit-pet/` level is tolerated, since torchvision creates
one. Then check it before running anything — this needs no dependencies:

```bash
python scripts/check_data.py --data $DATA
```

It should report 3680 / 3669 images for Pets and 5994 / 5794 for CUB, at 37 and
200 classes. Those are the standard splits every number in the thesis was
computed on; if your counts differ, so will your results.

## Checking a number

No GPU needed — every figure is built from `results/results.csv`:

```bash
pip install pandas matplotlib
python scripts/f1_grid.py     # Figure 1, prints the best cell per dataset
python scripts/f2_prompts.py  # Figure 2
python scripts/f3_cfg.py      # Figure 3
```

`f1_grid.py` should report `up_1` at t=100 for Pets and `down_2` for CUB, at 66.8
and 23.4.

## Regenerating it

```bash
python src/extract_baselines.py --prime    
sbatch slurm/e1_extract.slurm pets 100     # one grid cell, both splits
sbatch slurm/probe.slurm                   # probes whatever is new
```

`probe.slurm` skips anything already in the csv, so it is safe to rerun.
`e3_queue.sh`, `e4_queue.sh` and `e4_p2_extension.sh` submit the prompt-ladder and
guidance sweeps in bulk.

Every cluster-specific path lives in `slurm/env.sh`, which the job scripts source;
the partitions in the `#SBATCH` headers are the only other ALICE-specific part. The
`src/` scripts underneath take plain arguments and have no cluster dependency.

## Adapters

```bash
python src/extract_features.py --dataset pets --split test --t 200 \
    --prompt P3 --lora lora/pets_r16/ckpt-5000
```

`scripts/lora_stats.py` measures what training did to the weights;
`scripts/lora_figure.py` draws it.

## results.csv

| column | |
|---|---|
| `dataset` | `pets` or `cub` |
| `t` | noise level, `-` for the reference encoders |
| `prompt` | `P0`–`P4`; `P3`/`P4` name the class and are oracles |
| `cfg` | guidance scale, or `off` |
| `lora` | adapter path, or `none` |
| `block` | UNet block. `cond/` and `cfg/` mark the two branches of a guidance run; a model name (`dinov2`, `clip`, `resnet50`) marks a reference encoder |
| `seed` | probe seed, 0–2 |
| `probe_acc`, `knn_acc` | top-1 as a fraction |
| `n_train`, `n_test` | split sizes |
| `train_file` | source feature file |

Duplicated extractions agree to within 0.2 points. Treat smaller differences as
ties.

## Environment

Python 3.11. `requirements.txt` is a freeze of the environment everything was run
in — torch 2.5.1+cu121, diffusers 0.39, transformers 5.14, peft 0.20:

```bash
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

The extra index is needed because the `+cu121` torch wheels are not on PyPI.

Jobs run with `HF_HUB_OFFLINE=1`; compute nodes never download, so weights are
primed on a login node first.
