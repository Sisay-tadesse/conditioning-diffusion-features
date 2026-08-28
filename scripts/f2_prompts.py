"""F2: the prompt ladder at the best block, two noise levels each."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, OUT

PROMPTS = ["P0", "P1", "P2", "P3", "P4"]
CFG_T = {"pets": {"block": "up_1", "ts": [100, 200]}, "cub": {"block": "down_2", "ts": [200, 300]}}

df = load()
e3 = df[(df.cfg == "off") & (~df.is_lora) & (df.prompt.isin(PROMPTS))]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True, sharex=True)
for ax, ds in zip(axes, ["pets", "cub"]):
    cfg = CFG_T[ds]
    for t, marker in zip(cfg["ts"], ["o", "s"]):
        sub = e3[(e3.dataset == ds) & (e3.block == cfg["block"]) & (e3.t == t)]
        acc = sub.groupby("prompt")["probe_acc"].mean().reindex(PROMPTS)
        ax.plot(PROMPTS, acc.values, marker=marker, label=f"$t={t}$")
        print(ds, t, {p: round(v, 1) for p, v in acc.items()})
    ax.axvspan(2.5, 4.5, color="0.85", zorder=0)  # P3/P4: oracle
    ax.set_title({"pets": "Pets @ up_1", "cub": "CUB @ down_2"}[ds], fontsize=10)
    ax.set_xlabel("prompt")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("probe acc. (%)")
for ax in axes:
    ymin, ymax = ax.get_ylim()
    ax.text(3.5, ymin + 0.04 * (ymax - ymin), "oracle (label leak)", ha="center", fontsize=8, color="0.35")
fig.savefig(OUT / "f2_prompts.pdf")
print("wrote f2_prompts.pdf")
