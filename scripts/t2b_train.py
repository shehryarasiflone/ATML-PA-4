"""Task 2 Part B: train the MQAR models (B1, B2, B3) for seeds 6304-6306.

  python scripts/t2b_train.py --runs B1          # dense K4/8/16 mixed, dense K16 near, window K4/8/16 near, window K16 mixed
  python scripts/t2b_train.py --runs B2          # linear K4/8/16 mixed, dilated K4/16 mixed
  python scripts/t2b_train.py --runs B3          # intervention (hybrid) K16 mixed
Each run: results/task2/mqar/runs/<run>_s<seed>/{curve.csv, dev_evals.json, test.json, test_correct.npz, model.pt}
Finished runs are skipped. Dense models are also evaluated with a W=64 mask imposed at test time only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src.common.utils import RESULTS_DIR, save_json, set_seed
from src.task2.mqar import (MQARModel, batch_to, evaluate, load_eval, loss_acc, lr_at, n_params, train_stream)

RUNS = {
    "B1": {"dense_K4_mixed": ("dense", 4, "mixed"), "dense_K8_mixed": ("dense", 8, "mixed"),
           "dense_K16_mixed": ("dense", 16, "mixed"), "dense_K16_near": ("dense", 16, "near"),
           "window_K4_near": ("window", 4, "near"), "window_K8_near": ("window", 8, "near"),
           "window_K16_near": ("window", 16, "near"), "window_K16_mixed": ("window", 16, "mixed")},
    "B2": {"linear_K4_mixed": ("linear", 4, "mixed"), "linear_K8_mixed": ("linear", 8, "mixed"),
           "linear_K16_mixed": ("linear", 16, "mixed"), "dilated_K4_mixed": ("dilated", 4, "mixed"),
           "dilated_K16_mixed": ("dilated", 16, "mixed")},
    "B3": {"hybrid_K16_mixed": ("hybrid", 16, "mixed")},
}
CKPTS = [500, 1000, 1500, 2000, 3000, 4500]
STEPS, BS = 4500, 32
OUT = RESULTS_DIR / "task2" / "mqar" / "runs"

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", default=["B1"])
p.add_argument("--seeds", nargs="+", type=int, default=[6304, 6305, 6306])
p.add_argument("--hybrid-W", type=int, default=34)   # state bytes matched to linear attention, see t2b_costs.py
args = p.parse_args()
dev = "cuda"
test_sets, dev_sets = load_eval("test"), load_eval("development")

todo = {}
for r in args.runs:
    todo.update(RUNS[r] if r in RUNS else {r: next(v[r] for v in RUNS.values() if r in v)})

for name, (kind, K, reg) in todo.items():
    for seed in args.seeds:
        d = OUT / f"{name}_s{seed}"
        if (d / "test.json").exists():
            print(f"[{name} s{seed}] done, skipping", flush=True)
            continue
        d.mkdir(parents=True, exist_ok=True)
        data = train_stream(K, reg, seed)
        set_seed(seed)
        model = MQARModel(kind, hybrid_W=args.hybrid_W if kind == "hybrid" else None).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
        curve, devs, buf = [], {}, []
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for step in range(STEPS):
            for gr in opt.param_groups:
                gr["lr"] = lr_at(step)
            t, q, a, _ = batch_to(data, slice(step * BS, (step + 1) * BS), dev)
            loss, acc, _, _ = loss_acc(model, t, q, a)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            buf.append((loss.item(), acc.item(), gn.item()))
            if (step + 1) % 25 == 0:
                b = np.array(buf)
                curve.append({"update": step + 1, "lr": lr_at(step), "train_loss": b[:, 0].mean(),
                              "train_acc": b[:, 1].mean(), "grad_norm": b[:, 2].mean()})
                buf = []
            if (step + 1) in CKPTS:
                devs[step + 1] = evaluate(model, dev_sets, dev)
                own = devs[step + 1][f"K{K}_{reg}"]
                print(f"[{name} s{seed}] update {step + 1}: train loss {curve[-1]['train_loss']:.3f} "
                      f"| dev {reg} K{K} acc {own['acc']:.3f}", flush=True)
        torch.cuda.synchronize()
        train_sec = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**20
        res, cor = evaluate(model, test_sets, dev, keep_correct=True)
        out = {"run": name, "seed": seed, "attention": kind, "train_K": K, "train_regime": reg,
               "params": n_params(model), "train_sec": round(train_sec, 1), "ms_per_update": 1000 * train_sec / STEPS,
               "train_peak_alloc_mb": round(peak, 1), "test": res}
        if kind == "hybrid":
            out["hybrid_W"] = args.hybrid_W
        npz = {f"{k}_correct": v[0] for k, v in cor.items()} | {f"{k}_dist": v[1] for k, v in cor.items()}
        if kind == "dense":                       # isolation control: W=64 mask at test time only
            res_w, cor_w = evaluate(model, test_sets, dev, kind="window", keep_correct=True)
            out["test_window_mask_at_test"] = res_w
            npz |= {f"tw_{k}_correct": v[0] for k, v in cor_w.items()}
        np.savez_compressed(d / "test_correct.npz", **npz)
        pd.DataFrame(curve).to_csv(d / "curve.csv", index=False)
        save_json({str(k): v for k, v in devs.items()}, d / "dev_evals.json")
        torch.save(model.state_dict(), d / "model.pt")
        save_json(out, d / "test.json")
        print(f"[{name} s{seed}] {train_sec:.0f}s | test mixed acc " +
              " ".join(f"K{k}:{res[f'K{k}_mixed']['acc']:.3f}" for k in (4, 8, 16)) +
              f" | near K{K}: {res[f'K{K}_near']['acc']:.3f} far K{K}: {res[f'K{K}_far']['acc']:.3f}", flush=True)
print("All requested MQAR runs done.", flush=True)
