"""F1: probe accuracy over t x block, null prompt, no guidance, seed-mean."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import load, OUT

BLOCKS = ["down_2", "mid", "up_0", "up_1", "up_2"]
TS = [50, 100, 200, 300, 500]

df = load()
e1 = df[(df.prompt == "P0") & (df.cfg == "off") & (~df.is_lora) & (df.block.isin(BLOCKS))]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
for ax, ds in zip(axes, ["pets", "cub"]):
    sub = e1[e1.dataset == ds].groupby(["t", "block"])["probe_acc"].mean().unstack()
    sub = sub.reindex(index=TS, columns=BLOCKS)
    m = sub.to_numpy()
    im = ax.imshow(m, cmap="viridis", aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if not np.isnan(m[i, j]):
                peak = m[i, j] == np.nanmax(m)
                ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        fontweight="bold" if peak else "normal",
                        color="white" if m[i, j] < np.nanmax(m) * 0.75 else "black")
    ax.set_xticks(range(len(BLOCKS)), BLOCKS, fontsize=8)
    ax.set_yticks(range(len(TS)), TS, fontsize=8)
    ax.set_xlabel("UNet block")
    ax.set_title({"pets": "Oxford-IIIT Pets", "cub": "CUB-200"}[ds], fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85, label="probe acc. (%)")
axes[0].set_ylabel("noise level $t$")
fig.savefig(OUT / "f1_grid.pdf")
print("wrote f1_grid.pdf")
# print the winning cell, to check against the caption
for ds in ["pets", "cub"]:
    sub = e1[e1.dataset == ds].groupby(["t", "block"])["probe_acc"].mean()
    print(ds, "max:", sub.idxmax(), round(sub.max(), 1))
