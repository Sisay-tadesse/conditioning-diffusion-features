"""Are the datasets where the loaders expect, with the right splits?

Reads the same index files src/extract_features.py does, but with the standard
library only, so it works before the environment is installed.

    python scripts/check_data.py --data ~/datasets
"""
import argparse
import os
import sys
from pathlib import Path

EXPECTED = {("pets", "trainval"): (3680, 37), ("pets", "test"): (3669, 37),
            ("cub", "train"): (5994, 200), ("cub", "test"): (5794, 200)}


def pets(root, split):
    root = Path(root) / "pets"
    if not (root / "annotations").exists():
        root = root / "oxford-iiit-pet"
    lines = (root / "annotations" / f"{split}.txt").read_text().split("\n")
    rows = [l.split() for l in lines if l.strip()]
    missing = [r[0] for r in rows[:50] if not (root / "images" / f"{r[0]}.jpg").exists()]
    return len(rows), len({int(r[1]) for r in rows}), missing


def cub(root, split):
    root = Path(root) / "cub" / "CUB_200_2011"
    ids = dict(l.split() for l in (root / "images.txt").read_text().split("\n") if l.strip())
    flag = dict(l.split() for l in
                (root / "train_test_split.txt").read_text().split("\n") if l.strip())
    want = "1" if split == "train" else "0"
    keep = [i for i in ids if flag[i] == want]
    n_cls = len([l for l in (root / "classes.txt").read_text().split("\n") if l.strip()])
    missing = [ids[i] for i in keep[:50] if not (root / "images" / ids[i]).exists()]
    return len(keep), n_cls, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("DATA",
                    os.path.expanduser("~/datasets")))
    args = ap.parse_args()

    bad = 0
    for (ds, split), (n_exp, k_exp) in EXPECTED.items():
        try:
            n, k, missing = (pets if ds == "pets" else cub)(args.data, split)
        except OSError as e:
            print(f"FAIL {ds}/{split}: {e}")
            bad += 1
            continue
        ok = n == n_exp and k == k_exp and not missing
        print(f"{'ok  ' if ok else 'BAD '} {ds}/{split}: {n} images, {k} classes"
              + ("" if n == n_exp and k == k_exp else f"  expected {n_exp}, {k_exp}")
              + (f"  missing e.g. {missing[0]}" if missing else ""))
        bad += not ok
    print("all good" if not bad else f"{bad} problem(s) — see the layout in the README")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
