"""F3: accuracy vs guidance scale, oracle and deployable arms, LoRA overlaid."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, OUT

CONF = {"pets": {"t": 200, "cfgb": "cfg/up_1", "condb": "cond/up_1", "p0": None},
        "cub": {"t": 300, "cfgb": "cfg/down_2", "condb": "cond/down_2", "p0": None}}

df = load()
# the no-conditioning line each curve has to beat
for ds, c in CONF.items():
    ref = df[(df.dataset == ds) & (df.prompt == "P0") & (df.cfg == "off") & (~df.is_lora)
             & (df.block == c["cfgb"].split("/")[1]) & (df.t == c["t"])]
    c["p0"] = ref["probe_acc"].mean()

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
for ax, ds in zip(axes, ["pets", "cub"]):
    c = CONF[ds]
    base = df[(df.dataset == ds) & (df.t == c["t"]) & (~df.is_lora)]
    for prompt, color, label in [("P3", "tab:red", "P3 (oracle)"), ("P2", "tab:blue", "P2 (deployable)")]:
        sub = base[(base.prompt == prompt) & (base.block == c["cfgb"])]
        acc = sub.groupby("cfg_w")["probe_acc"].mean().sort_index()
        ax.plot(acc.index, acc.values, marker="o", color=color, label=label)
        print(ds, prompt, "cfg:", {w: round(v, 1) for w, v in acc.items()})
        cond = base[(base.prompt == prompt) & (base.block == c["condb"])]
        if len(cond):
            cacc = cond["probe_acc"].mean()
            ax.axhline(cacc, color=color, ls=":", lw=1, alpha=0.7)
            print(ds, prompt, "cond mean:", round(cacc, 1))
    # does the guidance gain survive adaptation?
    lora = df[(df.dataset == ds) & (df.t == c["t"]) & df.is_lora & (df.prompt == "P2")
              & (df.block == c["cfgb"])]
    if len(lora):
        lacc = lora.groupby("cfg_w")["probe_acc"].mean()
        ax.scatter(lacc.index, lacc.values, marker="x", s=60, color="black", zorder=5,
                   label="P2 + LoRA")
        print(ds, "lora pts:", {w: round(v, 1) for w, v in lacc.items()})
    ax.axhline(c["p0"], color="0.4", ls="--", lw=1, label="P0, no CFG")
    ax.set_title({"pets": "Pets, cfg/up_1, $t{=}200$", "cub": "CUB, cfg/down_2, $t{=}300$"}[ds],
                 fontsize=10)
    ax.set_xlabel("guidance scale $s$")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("probe acc. (%)")
axes[0].legend(fontsize=8, loc="center right")
fig.savefig(OUT / "f3_cfg.pdf")
print("wrote f3_cfg.pdf")
