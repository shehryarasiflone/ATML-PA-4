"""Task 1, steps 2, 3 and 8: P1 magnitude and P2 Wanda pruning at 25/50/75%, plus the 2:4 diagnostic.

Examples (one process per GPU):
  python scripts/t1_prune.py --methods magnitude --baseline-dev
  python scripts/t1_prune.py --methods wanda --nm
Each condition starts from a fresh copy of the FP16 checkpoint and is skipped if already finished.
Outputs: results/task1/conditions/<tag>/{summary.json, layer_records.csv, ppl_*.csv, hellaswag_*.csv}
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.common.utils import SEED, load_json, save_json, set_seed
from src.task1.calib import sequential_pass
from src.task1.evaluate import T1_DIR, dev_perplexity, evaluate_condition, load_calibration, load_hellaswag
from src.task1.model_utils import load_base, load_tokenizer, param_accounting
from src.task1.prune import (magnitude_score, mask_overlap, nm_prune_mask, row_prune_mask, wanda_score)
from src.task1.storage import dense_fp16_bytes, nm24_bytes, sparse_mask_bytes

p = argparse.ArgumentParser()
p.add_argument("--methods", nargs="+", default=["magnitude", "wanda"], choices=["magnitude", "wanda"])
p.add_argument("--sparsities", nargs="+", type=float, default=[0.25, 0.50, 0.75])
p.add_argument("--nm", action="store_true", help="also run the 2:4 Wanda diagnostic")
p.add_argument("--baseline-dev", action="store_true", help="add C4-dev perplexity to the FP16 baseline")
p.add_argument("--bs", type=int, default=8)
args = p.parse_args()

set_seed(SEED)
tok = load_tokenizer()
calib, calib_rows = load_calibration("calibration")
hs_all = load_hellaswag(tok)
print("calibration:", tuple(calib.shape), flush=True)

if args.baseline_dev:
    f = T1_DIR / "conditions" / "fp16_baseline" / "summary.json"
    s = load_json(f)
    if "c4dev" not in s:
        m = load_base()
        s["c4dev"] = dev_perplexity(m)
        save_json(s, f)
        print("baseline c4dev ppl:", round(s["c4dev"]["ppl"], 3), flush=True)
        del m
        torch.cuda.empty_cache()


def run(tag, score_fn, mask_fn, hs, storage_fn, meta):
    d = T1_DIR / "conditions" / tag
    if (d / "summary.json").exists():
        print(f"[{tag}] already done, skipping", flush=True)
        return
    print(f"\n===== {tag} =====", flush=True)
    model = load_base()
    torch.cuda.reset_peak_memory_stats()

    def method(li, lin, st):
        from src.task1.calib import GROUP_OF
        out = {}
        for pname, mod in lin.items():
            g = st[GROUP_OF[pname]]
            W = mod.weight.data
            score = score_fn(W, g)
            keep = mask_fn(score)
            rec = {"score_q50": float(score.median()), "act_norm_max_over_median":
                   float(g["act_norm"].max() / g["act_norm"].median())}
            if score_fn is wanda_score:           # how differently would magnitude have pruned?
                rec["overlap_with_magnitude"] = mask_overlap(keep, mask_fn(magnitude_score(W)))
            W.mul_(keep.to(W.dtype))
            out[pname] = rec
        return out

    recs, sec = sequential_pass(model, calib, method, bs=args.bs)
    comp_peak = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    df = pd.DataFrame(recs)
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "layer_records.csv", index=False)
    acc = param_accounting(model)
    stor = storage_fn(model)
    print(f"[{tag}] eligible sparsity achieved: {1 - acc['eligible_nonzero_fraction']:.6f} "
          f"| compression {sec:.0f}s | mean rel. output error {df.rel_out_err.mean():.4f}", flush=True)
    extra = {**meta, "achieved_eligible_sparsity": 1 - acc["eligible_nonzero_fraction"], "params": acc,
             "compression_sec": round(sec, 1), "compression_peak_alloc_gb": comp_peak,
             "storage": {"dense_fp16_bytes": dense_fp16_bytes(model), **{k: v for k, v in stor.items()}},
             "calib_rel_out_err_mean": float(df.rel_out_err.mean()),
             "calib_rel_out_err_by_proj": df.groupby("proj").rel_out_err.mean().to_dict()}
    evaluate_condition(model, tok, tag, hs=hs, hs_cache=hs_all, extra=extra)
    del model
    torch.cuda.empty_cache()


for method in args.methods:
    for s in args.sparsities:
        tag = f"{'P1_magnitude' if method == 'magnitude' else 'P2_wanda'}_s{int(round(s * 100))}"
        run(tag, magnitude_score if method == "magnitude" else wanda_score,
            lambda sc, s=s: row_prune_mask(sc, s), "full" if abs(s - 0.5) < 1e-9 else "diag",
            sparse_mask_bytes, {"method": method, "target_sparsity": s, "mask": "per-output-row unstructured"})

if args.nm:
    run("P2_wanda_2of4", wanda_score, nm_prune_mask, "full",
        lambda m: {**nm24_bytes(m), **{f"mask_format_{k}": v for k, v in sparse_mask_bytes(m).items()}},
        {"method": "wanda", "target_sparsity": 0.5, "mask": "2:4 along input columns"})
print("\nAll requested pruning conditions done.", flush=True)
