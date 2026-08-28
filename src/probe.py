"""Linear probe + k-NN on cached features, one row per config into results.csv.

Deliberately weak head, so differences are the representation and not the
classifier. k-NN fits nothing, and moves with the probe when the structure is
really there.

    python src/probe.py --dir ~/features --results results/results.csv
"""
import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

CSV_FIELDS = ["timestamp", "dataset", "t", "prompt", "cfg", "lora", "block",
              "seed", "probe_acc", "knn_acc", "n_train", "n_test", "train_file"]

TRAIN_SPLITS = ("trainval", "train")  # pets, cub


def train_probe(xtr, ytr, xte, yte, n_classes, seed=0, epochs=100,
                lr=1e-3, batch=512, device="cpu"):
    torch.manual_seed(seed)
    # train statistics only
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-6
    xtr, xte = ((xtr - mu) / sd).to(device), ((xte - mu) / sd).to(device)
    ytr, yte = ytr.to(device), yte.to(device)

    lin = torch.nn.Linear(xtr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(xtr), device=device)
        for i in range(0, len(xtr), batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(lin(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        acc = (lin(xte).argmax(1) == yte).float().mean().item()
    return acc


@torch.no_grad()
def knn_acc(xtr, ytr, xte, yte, k=10, device="cpu"):
    xtr = F.normalize(xtr.to(device), dim=1)
    xte = F.normalize(xte.to(device), dim=1)
    ytr = ytr.to(device)
    preds = []
    for i in range(0, len(xte), 1024):        # chunked, the full matrix is big
        sim = xte[i:i + 1024] @ xtr.T
        nn_lab = ytr[sim.topk(k, dim=1).indices]          # [B, k]
        preds.append(torch.mode(nn_lab, dim=1).values)
    return (torch.cat(preds) == yte.to(device)).float().mean().item()


def probe_pair(train_path, test_path, results_path, seed, device, force=False):
    tr, te = torch.load(train_path, map_location="cpu", weights_only=True), \
             torch.load(test_path, map_location="cpu", weights_only=True)
    meta = tr.get("meta", {})
    n_classes = len(tr["classes"])

    done = set()
    if os.path.exists(results_path) and os.path.getsize(results_path) > 0 \
            and not force:
        with open(results_path) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "train_file" in reader.fieldnames:
                done = {(r["train_file"], r["block"], r["seed"])
                        for r in reader}

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    new_file = (not os.path.exists(results_path)
                or os.path.getsize(results_path) == 0)
    fname = os.path.basename(train_path)
    with open(results_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        for block in tr["feats"]:
            if (fname, block, str(seed)) in done:
                print(f"skip {fname} {block} (already in csv)")
                continue
            p_acc = train_probe(tr["feats"][block], tr["labels"],
                                te["feats"][block], te["labels"],
                                n_classes, seed=seed, device=device)
            k_acc = knn_acc(tr["feats"][block], tr["labels"],
                            te["feats"][block], te["labels"], device=device)
            row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "dataset": meta.get("dataset", "?"),
                   "t": meta.get("t", "?"),
                   "prompt": meta.get("prompt", "?"),
                   "cfg": meta.get("cfg") if meta.get("cfg") is not None else "off",
                   "lora": meta.get("lora") or "none",
                   "block": block, "seed": seed,
                   "probe_acc": f"{p_acc:.4f}", "knn_acc": f"{k_acc:.4f}",
                   "n_train": len(tr["labels"]), "n_test": len(te["labels"]),
                   "train_file": fname}
            w.writerow(row)
            f.flush()
            print(f"{fname} {block}: probe={p_acc:.4f} knn={k_acc:.4f}")


def find_pairs(feat_dir):
    """Pair train and test files: the names differ only in the split field."""
    files = sorted(Path(feat_dir).glob("*.pt"))
    names = {f.name for f in files}
    pairs = []
    for f in files:
        parts = f.name.split("_")
        if len(parts) < 3 or parts[1] not in TRAIN_SPLITS:
            continue
        if "_limit" in f.name:
            continue                       # smoke tests
        test_name = "_".join([parts[0], "test"] + parts[2:])
        if test_name in names:
            pairs.append((str(f), str(f.parent / test_name)))
        else:
            print(f"no test file yet for {f.name}")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="probe every pair in here")
    ap.add_argument("--train", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="redo rows already in csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    if args.train and args.test:
        pairs = [(args.train, args.test)]
    elif args.dir:
        pairs = find_pairs(os.path.expanduser(args.dir))
        print(f"{len(pairs)} train/test pairs found")
    else:
        ap.error("give --dir, or --train and --test")

    for train_path, test_path in pairs:
        probe_pair(train_path, test_path, args.results, args.seed,
                   device, force=args.force)
    print("all done -> " + args.results)


if __name__ == "__main__":
    main()
