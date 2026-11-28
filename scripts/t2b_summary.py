"""Print a compact Part B table (mean +- sd over seeds) from results/task2/mqar/runs/*/test.json.

  python scripts/t2b_summary.py
Also writes results/task2/mqar/summary.csv.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.utils import RESULTS_DIR

rows = []
for f in sorted((RESULTS_DIR / "task2" / "mqar" / "runs").glob("*/test.json")):
    r = json.loads(f.read_text())
    K = r["train_K"]
    for variant, res in [("trained mask", r["test"])] + ([("W=64 at test only", r["test_window_mask_at_test"])]
                                                         if "test_window_mask_at_test" in r else []):
        row = {"run": r["run"], "variant": variant, "seed": r["seed"], "ms_per_update": r["ms_per_update"]}
        for reg in ("near", "far", "mixed"):
            row[f"K{K}_{reg}"] = res[f"K{K}_{reg}"]["acc"]
        for k in (4, 8, 16):
            row[f"mixed_K{k}"] = res[f"K{k}_mixed"]["acc"]
        for b in ("0_32", "32_64", "64_128", "128_256"):
            row[f"K{K}_mixed_d{b}"] = res[f"K{K}_mixed"][f"acc_d{b}"]
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / "task2" / "mqar" / "summary.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
for (run, var), g in df.groupby(["run", "variant"], sort=False):
    K = [c for c in g.columns if c.endswith("_near") and g[c].notna().all()][0].split("_")[0]
    cols = [f"{K}_near", f"{K}_far", f"{K}_mixed"] + [f"{K}_mixed_d{b}" for b in ("0_32", "32_64", "64_128", "128_256")]
    s = " | ".join(f"{c.replace(K + '_', '')} {g[c].mean():.3f}+-{g[c].std():.3f}" for c in cols)
    seeds = " ".join(f"{v:.2f}" for v in g[f"{K}_mixed"])
    print(f"{run:18s} {var:18s} n={len(g)} {s} | per-seed mixed: {seeds} | {g.ms_per_update.mean():.1f} ms/upd")