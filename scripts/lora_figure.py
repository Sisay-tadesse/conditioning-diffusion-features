"""F4: three panels against the easy objections to the LoRA null result -
wrong layers, too small, too few steps. Pets shown; CUB looks the same.
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
STATS = ROOT / "results" / "lora_stats.csv"
BASE = ROOT / "results" / "lora_base_norms.csv"
OUT = ROOT / "figures"

SELF, CROSS, PLAIN, HI = "#7f9bb5", "#c0392b", "#b0b0b0", "#5b8c5a"

base = {r["layer"]: r for r in csv.DictReader(open(BASE))}
rows = [r for r in csv.DictReader(open(STATS))
        if int(r["step"]) == 5000 and r["ds"] == "pets"]
for r in rows:
    b = base[r["layer"][len("unet."):]]
    r["fro"] = float(r["fro"])
    r["base_fro"] = float(b["base_fro"])
    r["rho"] = r["fro"] / r["base_fro"]      # update size vs the weight

fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)

# (a) which projections moved
LAB = [("attn1", "to_q", "self-attention: query"),
       ("attn1", "to_k", "self-attention: key"),
       ("attn1", "to_v", "self-attention: value"),
       ("attn1", "to_out", "self-attention: output"),
       ("attn2", "to_q", "cross-attention: query"),
       ("attn2", "to_k", "cross-attention: key"),
       ("attn2", "to_v", "cross-attention: value"),
       ("attn2", "to_out", "cross-attention: output")]
v = [100 * np.mean([r["rho"] for r in rows if r["kind"] == k and r["proj"] == p])
     for k, p, _ in LAB]
y = np.arange(8)
ax[0].barh(y, v, color=[SELF if k == "attn1" else CROSS for k, _, _ in LAB], height=.72)
for i, val in enumerate(v):
    ax[0].text(val + .3, i, f"{val:.1f}%", va="center", fontsize=8.5, color="0.25")
ax[0].set_yticks(y)
ax[0].set_yticklabels([l for _, _, l in LAB], fontsize=8.5)
ax[0].invert_yaxis()
ax[0].set_xlim(0, 18.5)
ax[0].set_xlabel("how far the weights moved\n(% of their original size)", fontsize=9)
ax[0].set_title("The text pathway changed most", fontsize=10.5)

# (b) where in the network. Bigger weights take bigger updates, so take that
# trend out first.
lb = np.log([r["base_fro"] for r in rows])
lr = np.log([r["rho"] for r in rows])
slope, icpt = np.polyfit(lb, lr, 1)
res = lr - (slope * lb + icpt)
ORD = ["down_0", "down_1", "down_2", "mid", "up_1", "up_2", "up_3"]
READ = {"down_2", "mid", "up_1"}          # the blocks we probe
vb = [np.exp(np.mean([res[i] for i, r in enumerate(rows) if r["block"] == b]))
      for b in ORD]
ax[1].bar(np.arange(7), vb, color=[HI if b in READ else PLAIN for b in ORD], width=.72)
ax[1].axhline(1.0, color="0.25", ls="--", lw=1)
ax[1].text(6.45, 1.02, "network average", ha="right", fontsize=8, color="0.35")
for i, b in enumerate(ORD):
    if b in READ:
        ax[1].text(i, vb[i] + .04, "read", ha="center", fontsize=8,
                   color=HI, weight="bold")
ax[1].set_xticks(np.arange(7))
ax[1].set_xticklabels(ORD, rotation=30, fontsize=8.5)
ax[1].set_ylim(0, 1.72)
ax[1].set_ylabel("adaptation, relative to\nthe network average", fontsize=9)
ax[1].set_title("Strongest where we read features", fontsize=10.5)

# (c) same update getting bigger, or a different one? weighted by how much
# each layer moved, so tiny layers do not dominate.
w = np.array([r["fro"] for r in rows])
pairs = [(1000, 2500), (2500, 5000), (1000, 5000)]
vc = [float((np.array([float(r[f"cos_{a}_{b}"]) for r in rows]) * w).sum() / w.sum())
      for a, b in pairs]
print("cosines:", {f"{a}-{b}": round(c, 3) for (a, b), c in zip(pairs, vc)})
print("1k-2.5k x 2.5k-5k =", round(vc[0] * vc[1], 3), "vs measured 1k-5k", round(vc[2], 3))
ax[2].bar(np.arange(3), vc, color=["#d9a441", "#d9a441", "#b5761f"], width=.6)
for i, val in enumerate(vc):
    ax[2].text(i, val + .03, f"{val:.2f}", ha="center", fontsize=9, color="0.25")
ax[2].axhline(1.0, color="0.25", ls="--", lw=1)
ax[2].text(-0.42, 1.06, "1.0 would mean one fixed direction", fontsize=8, color="0.35")
ax[2].set_xticks(np.arange(3))
ax[2].set_xticklabels([f"{a/1000:g}k vs {b/1000:g}k" for a, b in pairs],
                      fontsize=8.5)
ax[2].set_ylim(0, 1.22)
ax[2].set_ylabel("how similar the change is\nbetween two checkpoints", fontsize=9)
ax[2].set_title("The direction keeps drifting", fontsize=10.5)

for a in ax:
    a.grid(alpha=.25, axis="x" if a is ax[0] else "y")
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "f4_lora.pdf")
fig.savefig(OUT / "f4_lora.png", dpi=190)
print("wrote f4_lora.pdf")
