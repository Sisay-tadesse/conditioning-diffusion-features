"""What the adapters did to the weights, measured from the checkpoints.

Written to rule out the easy explanations for the null probe result: too small
an update, wrong layers, rank 16 too tight.

    python scripts/lora_stats.py --lora lora --out results/lora_stats.csv
"""
import argparse
import csv
import os
import re

import numpy as np
from safetensors.numpy import load_file

DS = ["pets", "cub"]
STEPS = [1000, 2500, 5000]

# peft key -> block, attention, projection
DOWN_UP = re.compile(
    r"(down_blocks|up_blocks)\.(\d+)\.attentions\.(\d+)"
    r"\.transformer_blocks\.0\.(attn[12])\.(to_\w+(?:\.0)?)$")
MID = re.compile(
    r"mid_block\.attentions\.(\d+)\.transformer_blocks\.0"
    r"\.(attn[12])\.(to_\w+(?:\.0)?)$")


def parse(n):
    n = n[len("unet."):]
    m = DOWN_UP.match(n)
    if m:
        return dict(block=f"{'down' if m.group(1)[0] == 'd' else 'up'}_{m.group(2)}",
                    bi=int(m.group(2)), att=int(m.group(3)), kind=m.group(4),
                    proj=m.group(5).replace(".0", ""), side=m.group(1)[:4])
    m = MID.match(n)
    if m:
        return dict(block="mid", bi=-1, att=int(m.group(1)), kind=m.group(2),
                    proj=m.group(3).replace(".0", ""), side="mid")
    raise ValueError(n)


def load(root, ds, step):
    sd = load_file(f"{root}/{ds}_r16/ckpt-{step}/pytorch_lora_weights.safetensors")
    names = sorted({k.rsplit(".lora_", 1)[0] for k in sd})
    return {n: (sd[f"{n}.lora_A.weight"].astype(np.float64),
                sd[f"{n}.lora_B.weight"].astype(np.float64)) for n in names}


def factor(A, B):
    """dW = B @ A without building dW (it would be d_out x d_in)."""
    QB, RB = np.linalg.qr(B)        # (d_out, r), (r, r)
    QA, RA = np.linalg.qr(A.T)      # (d_in, r),  (r, r)
    return QB, RB @ RA.T, QA        # the middle (r, r) carries the geometry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="lora", help="holds <ds>_r16/ckpt-*/")
    ap.add_argument("--out", default="results/lora_stats.csv")
    args = ap.parse_args()

    cache = {(ds, st): load(args.lora, ds, st) for ds in DS for st in STEPS}

    rows = []
    for ds in DS:
        for n in sorted(cache[(ds, 5000)]):
            p = parse(n)
            for st in STEPS:
                A, B = cache[(ds, st)][n]
                _, M, _ = factor(A, B)
                s = np.linalg.svd(M, compute_uv=False)
                fro = float(np.sqrt((s ** 2).sum()))
                # stable rank: how many directions it really uses
                srank = float((s ** 2).sum() / s[0] ** 2) if s[0] > 0 else 0.0
                rows.append(dict(
                    ds=ds, step=st, layer=n, **p,
                    d_out=B.shape[0], d_in=A.shape[1],
                    fro=fro, rms=fro / np.sqrt(B.shape[0] * A.shape[1]),
                    srank=srank, s1=float(s[0]), s16=float(s[-1]),
                    s_ratio=float(s[-1] / s[0]) if s[0] > 0 else 0.0))

            # still turning, or settled? trace(A1' B1' B2 A2), no dW formed
            for a, b in [(1000, 2500), (2500, 5000), (1000, 5000)]:
                A1, B1 = cache[(ds, a)][n]
                A2, B2 = cache[(ds, b)][n]
                num = float(np.sum((B1.T @ B2) * (A2 @ A1.T).T))
                d1 = np.linalg.norm(B1 @ A1)
                d2 = np.linalg.norm(B2 @ A2)
                rows[-1][f"cos_{a}_{b}"] = num / (d1 * d2)

    keys = sorted({k for r in rows for k in r})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows, {len(rows) // (len(DS) * len(STEPS))} layers -> {args.out}")


if __name__ == "__main__":
    main()
