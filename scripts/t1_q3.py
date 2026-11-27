"""Task 1, step 7 (Q3): mixed-precision AWQ under the Q2 INT4 and INT3 packed-byte budgets.

Sensitivity: for every projection (unit = one matrix, 196 units) and every allowed width
b in {2, 3, 4, 8}, the relative calibration output error ||(W_q - W) X||^2 / ||W X||^2 after the same
AWQ-style scaling + quantizer as Q2 (alpha searched per group at that width), on full-precision
calibration activations (128 sequences).
Allocation: minimise sum_u cost_u(b_u) subject to packed bytes <= budget (multiple-choice knapsack,
greedy on the lower convex hull). Budget = the Q2 uniform model's total packed bytes (codes, FP16
scales, b-bit zero-points, serialized o_proj AWQ scales, and all FP16 non-eligible parameters).
Two cost definitions are compared on the C4 development set only (fixed in advance):
  A  relative output error (every matrix weighted equally)
  B  relative output error x matrix size / mean size (error per weight, larger matrices count more)
The lower development perplexity is the final Q3 allocation for that budget.

  python scripts/t1_q3.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.common.utils import SEED, load_json, save_json, set_seed
from src.task1.alloc import knapsack_bits
from src.task1.awq import awq_method, group_rel_errors, search_group
from src.task1.calib import GROUPS, sequential_pass
from src.task1.evaluate import T1_DIR, dev_perplexity, load_calibration, load_hellaswag
from src.task1.model_utils import eligible_shapes, load_base, load_tokenizer
from src.task1.quant import model_packed_bytes, packed_bytes
from src.task1.runner import done, finish, other_params

BITS = [2, 3, 4, 8]
OUT = T1_DIR / "q3"
OUT.mkdir(parents=True, exist_ok=True)
set_seed(SEED)
tok = load_tokenizer()
calib, _ = load_calibration("calibration")
elig = eligible_shapes()
shape_of = dict(elig)
n_o_scales = sum(i for (li, p), (o, i) in elig if p == "o_proj")


# 1. sensitivity -------------------------------------------------------------------
sf = OUT / "sensitivity.csv"
if sf.exists():
    sens = pd.read_csv(sf)
else:
    print("===== Q3 sensitivity: AWQ error per projection and bit width =====", flush=True)
    model = load_base()
    rows = []

    def method(li, lin, st, ctx):
        for g, projs in GROUPS.items():
            Ws = [lin[p].weight for p in projs]
            for b in BITS:
                a, s, _ = search_group(Ws, [b] * len(projs), st[g])
                for p, e in zip(projs, group_rel_errors(Ws, [b] * len(projs), s, st[g]["gram"])):
                    rows.append({"layer": li, "proj": p, "bits": b, "alpha": a, "rel_err": e})
        return None

    _, sec = sequential_pass(model, calib, method)
    other = other_params(model)
    del model
    torch.cuda.empty_cache()
    sens = pd.DataFrame(rows)
    sens.to_csv(sf, index=False)
    save_json({"sensitivity_sec": round(sec, 1), "other_params": other}, OUT / "sensitivity_meta.json")
other = load_json(OUT / "sensitivity_meta.json")["other_params"]


def total_bytes(bits_map):
    return model_packed_bytes([(o, i, bits_map[k]) for k, (o, i) in elig], other, n_o_scales)["total_bytes"]


budgets = {"int4budget": total_bytes({k: 4 for k, _ in elig}), "int3budget": total_bytes({k: 3 for k, _ in elig})}
fixed = total_bytes({k: 0 for k, _ in elig}) - sum(packed_bytes(o, i, 0) for _, (o, i) in elig)
mean_numel = sum(o * i for _, (o, i) in elig) / len(elig)


def allocate(name, budget):
    units = []
    for (li, p), (o, i) in elig:
        r = sens[(sens.layer == li) & (sens.proj == p)].set_index("bits").rel_err
        w = 1.0 if name == "A" else (o * i) / mean_numel
        units.append({"key": (li, p), "options": {b: (packed_bytes(o, i, b), w * float(r[b])) for b in BITS}})
    bits, used = knapsack_bits(units, budget - fixed)
    assert total_bytes(bits) <= budget
    return bits


# 2. candidates on the development set ------------------------------------------------
cf = OUT / "candidates.csv"
cands = pd.read_csv(cf).to_dict("records") if cf.exists() else []
for bname, budget in budgets.items():
    for cost in ["A", "B"]:
        if any(c["budget"] == bname and c["cost"] == cost for c in cands):
            continue
        bits = allocate(cost, budget)
        model = load_base()
        m, post = awq_method(lambda li, p, bits=bits: bits[(li, p)])
        sequential_pass(model, calib, m, post=post)
        dp = dev_perplexity(model)["ppl"]
        del model
        torch.cuda.empty_cache()
        cnt = pd.Series(list(bits.values())).value_counts().to_dict()
        cands.append({"budget": bname, "cost": cost, "dev_ppl": dp, "bytes": total_bytes(bits), "budget_bytes": budget,
                      **{f"n_b{b}": int(cnt.get(b, 0)) for b in BITS}})
        print(f"candidate {bname} cost {cost}: dev ppl {dp:.3f} | bits {cnt}", flush=True)
        pd.DataFrame(cands).to_csv(cf, index=False)

# 3. final Q3 runs -----------------------------------------------------------------------
hs_all = load_hellaswag(tok)
for bname, budget in budgets.items():
    tag = f"Q3_mixed_{bname}"
    c = pd.DataFrame(cands)
    c = c[c.budget == bname].sort_values("dev_ppl").iloc[0]
    bits = allocate(c.cost, budget)
    save_json({"budget": bname, "budget_bytes": budget, "bytes": total_bytes(bits), "cost": c.cost,
               "selected_by": "lowest C4-dev perplexity between cost A and B",
               "bits": {f"{li}.{p}": b for (li, p), b in bits.items()}}, OUT / f"allocation_{bname}.json")
    if done(tag):
        print(f"[{tag}] already done, skipping", flush=True)
        continue
    print(f"\n===== {tag} (cost {c.cost}) =====", flush=True)
    model = load_base()
    torch.cuda.reset_peak_memory_stats()
    m, post = awq_method(lambda li, p, bits=bits: bits[(li, p)])
    recs, sec = sequential_pass(model, calib, m, post=post)
    peak = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    stor = model_packed_bytes([(o, i, bits[k]) for k, (o, i) in elig], other, n_o_scales)
    stor["budget_bytes"] = budget
    stor["effective_bits_per_eligible_weight"] = 8 * stor["eligible_packed_bytes"] / sum(o * i for _, (o, i) in elig)
    finish(model, tok, tag, recs, sec, peak, stor,
           {"method": "AWQ mixed precision", "budget": bname, "cost": c.cost, "allowed_bits": BITS}, "full", hs_all)
print("\nQ3 done.", flush=True)
