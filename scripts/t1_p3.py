"""Task 1, step 4 (P3): nonuniform Wanda pruning with OWL-style block allocation.

Sensitivity: each block's outlier ratio D_l = share of its Wanda scores |W_ij|*||X_j|| above M times
the block mean (Yin et al., 2024), measured on the dense model with the 128 calibration sequences.
Allocation: keep-ratio linear in D_l over [1-S-lam, 1-S+lam], re-centred to mean 1-S; unit = whole
block (all 7 projections share s_l); exact zero-count matching to P1/P2 up to integer rounding.
Selection (fixed in advance): the (M, lam) candidate with the lowest C4-development perplexity
(32 sequences), ties -> smaller lam. No WikiText/C4-validation/HellaSwag numbers are used.

  python scripts/t1_p3.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src.common.utils import SEED, load_json, save_json, set_seed
from src.task1.alloc import owl_sparsities, row_counts, row_prune_mask_counts, zero_counts
from src.task1.calib import sequential_pass
from src.task1.evaluate import T1_DIR, dev_perplexity, load_calibration, load_hellaswag
from src.task1.model_utils import eligible_shapes, load_base, load_tokenizer
from src.task1.prune import wanda_score
from src.task1.runner import done, finish
from src.task1.storage import sparse_mask_bytes

MS, LAMS, TARGETS = [5, 7], [0.04, 0.08], [0.50, 0.75]
OUT = T1_DIR / "p3"
OUT.mkdir(parents=True, exist_ok=True)
set_seed(SEED)
tok = load_tokenizer()
calib, _ = load_calibration("calibration")


# 1. sensitivity on the dense model --------------------------------------------
@torch.no_grad()
def outlier_stats():
    f = OUT / "owl_stats.json"
    if f.exists():
        return load_json(f)
    print("===== OWL outlier statistics (dense model) =====", flush=True)
    model = load_base()
    from src.task1.calib import GROUP_OF
    rows = []

    def method(li, lin, st, ctx):
        mets = {p: wanda_score(m.weight.data, st[GROUP_OF[p]]) for p, m in lin.items()}
        total = sum(x.numel() for x in mets.values())
        mean = sum(float(x.sum()) for x in mets.values()) / total
        for M in [3, 5, 7, 10]:
            cnt = {p: int((x > M * mean).sum()) for p, x in mets.items()}
            rows.append({"layer": li, "M": M, "block_ratio": sum(cnt.values()) / total,
                         **{f"{p}_ratio": cnt[p] / mets[p].numel() for p in mets}})
        return None

    sequential_pass(model, calib, method, need_gram=False)
    del model
    torch.cuda.empty_cache()
    save_json(rows, f)
    return rows


stats = pd.DataFrame(outlier_stats())
# eligible shapes in layer order
elig = eligible_shapes()
N = sum(o * i for _, (o, i) in elig)


def allocation(target, M, lam):
    r = stats[stats.M == M].sort_values("layer").block_ratio.values
    s_block = owl_sparsities(r, target, lam)
    z = zero_counts([(o * i, s_block[li]) for (li, _), (o, i) in elig], int(round(target * N)))
    return s_block, {(li, p): int(zz) for ((li, p), _), zz in zip(elig, z)}


def prune(model, zmap, need_gram):
    from src.task1.calib import GROUP_OF

    def method(li, lin, st, ctx):
        for p, mod in lin.items():
            W = mod.weight.data
            keep = row_prune_mask_counts(wanda_score(W, st[GROUP_OF[p]]), row_counts(zmap[(li, p)], W.shape[0]))
            W.mul_(keep.to(W.dtype))
        return None
    return sequential_pass(model, calib, method, need_gram=need_gram)


# 2. candidates on the development set -------------------------------------------
cf = OUT / "candidates.csv"
cands = pd.read_csv(cf).to_dict("records") if cf.exists() else []
for S in TARGETS:
    f2 = T1_DIR / "conditions" / f"P2_wanda_s{int(S * 100)}" / "summary.json"
    p2 = load_json(f2) if f2.exists() else {"c4dev": {"ppl": float("nan")}}
    if f2.exists() and not any(c["target"] == S and c["lam"] == 0 for c in cands):
        cands.append({"target": S, "M": 0, "lam": 0.0, "dev_ppl": p2["c4dev"]["ppl"], "s_min": S, "s_max": S,
                      "note": "uniform P2 reference"})
    for M in MS:
        for lam in LAMS:
            if any(c["target"] == S and c["M"] == M and c["lam"] == lam for c in cands):
                continue
            s_block, zmap = allocation(S, M, lam)
            model = load_base()
            prune(model, zmap, need_gram=False)
            dp = dev_perplexity(model)["ppl"]
            del model
            torch.cuda.empty_cache()
            cands.append({"target": S, "M": M, "lam": lam, "dev_ppl": dp, "s_min": float(s_block.min()),
                          "s_max": float(s_block.max()), "note": "candidate"})
            print(f"candidate S={S} M={M} lam={lam}: dev ppl {dp:.3f} (uniform {p2['c4dev']['ppl']:.3f})", flush=True)
            pd.DataFrame(cands).to_csv(cf, index=False)
pd.DataFrame(cands).to_csv(cf, index=False)

# 3. final P3 runs with the selected candidate ------------------------------------
hs_all = load_hellaswag(tok)
for S in TARGETS:
    tag = f"P3_owl_s{int(S * 100)}"
    c = pd.DataFrame(cands)
    c = c[(c.target == S) & (c.lam > 0)].sort_values(["dev_ppl", "lam"]).iloc[0]
    M, lam = int(c.M), float(c.lam)
    s_block, zmap = allocation(S, M, lam)
    save_json({"target": S, "M": M, "lam": lam, "selected_by": "lowest C4-dev perplexity among lam>0 candidates",
               "block_sparsity": s_block.tolist(),
               "zeros_total": sum(zmap.values()), "zeros_target": int(round(S * N)), "eligible": N,
               "zeros_by_unit": {f"{li}.{p}": z for (li, p), z in zmap.items()}}, OUT / f"allocation_s{int(S * 100)}.json")
    if done(tag):
        print(f"[{tag}] already done, skipping", flush=True)
        continue
    print(f"\n===== {tag} (M={M}, lam={lam}) =====", flush=True)
    model = load_base()
    torch.cuda.reset_peak_memory_stats()
    recs, sec = prune(model, zmap, need_gram=True)
    peak = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    finish(model, tok, tag, recs, sec, peak, sparse_mask_bytes(model),
           {"method": "wanda + OWL block allocation", "target_sparsity": S, "M": M, "lam": lam,
            "mask": "per-output-row unstructured, block-dependent rate"},
           "full" if S == 0.5 else "diag", hs_all)
print("\nP3 done.", flush=True)
