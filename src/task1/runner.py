"""Shared helpers for running and saving one compression condition."""
import pandas as pd
import torch

from src.task1.evaluate import T1_DIR, evaluate_condition
from src.task1.model_utils import eligible_linears, param_accounting, unique_params
from src.task1.storage import dense_fp16_bytes


def done(tag):
    return (T1_DIR / "conditions" / tag / "summary.json").exists()


def other_params(model):
    ptrs = {m.weight.data_ptr() for _, m in eligible_linears(model)}
    return sum(p.numel() for _, p in unique_params(model) if p.data_ptr() not in ptrs)


def shapes(model):
    from src.task1.model_utils import layer_and_type
    return [(layer_and_type(n), tuple(m.weight.shape)) for n, m in eligible_linears(model)]


def finish(model, tok, tag, recs, sec, comp_peak_gb, storage, meta, hs, hs_cache):
    d = T1_DIR / "conditions" / tag
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(recs)
    df.to_csv(d / "layer_records.csv", index=False)
    acc = param_accounting(model)
    extra = {**meta, "params": acc, "achieved_eligible_sparsity": 1 - acc["eligible_nonzero_fraction"],
             "compression_sec": round(sec, 1), "compression_peak_alloc_gb": comp_peak_gb,
             "storage": {"dense_fp16_bytes": dense_fp16_bytes(model), **storage}}
    if "rel_out_err" in df:
        extra["calib_rel_out_err_mean"] = float(df.rel_out_err.mean())
        extra["calib_rel_out_err_by_proj"] = df.groupby("proj").rel_out_err.mean().to_dict()
    print(f"[{tag}] compression {sec:.0f}s | mean rel. output error {extra.get('calib_rel_out_err_mean', float('nan')):.4f}"
          f" | storage {storage.get('total_bytes', 0) / 1e6:.1f} MB", flush=True)
    s = evaluate_condition(model, tok, tag, hs=hs, hs_cache=hs_cache, extra=extra)
    del model
    torch.cuda.empty_cache()
    return s
