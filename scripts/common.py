"""Shared loader. Every figure and table comes out of results/results.csv."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "results.csv"
OUT = ROOT / "figures"


def load():
    df = pd.read_csv(CSV)
    # stored as fractions
    df["probe_acc"] = df["probe_acc"].astype(float) * 100
    df["knn_acc"] = df["knn_acc"].astype(float) * 100
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df["cfg_w"] = pd.to_numeric(df["cfg"], errors="coerce")  # NaN where cfg is "off"
    df["is_lora"] = df["lora"].ne("none")
    OUT.mkdir(exist_ok=True)
    return df
