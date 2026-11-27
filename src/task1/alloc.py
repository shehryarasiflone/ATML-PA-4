"""Allocation rules: exact zero-count bookkeeping for nonuniform pruning, OWL-style block sparsities,
and a byte-constrained bit allocation (multiple-choice knapsack, greedy on the lower convex hull)."""
import numpy as np
import torch


def owl_sparsities(outlier_ratio, target, lam, lo=0.10, hi=0.85):
    """Yin et al. (2024): keep-ratio grows linearly with the block's outlier ratio, spread over
    [1-target-lam, 1-target+lam] and re-centred so the mean equals 1 - target (blocks have equal size)."""
    r = np.asarray(outlier_ratio, dtype=float)
    span = r.max() - r.min()
    keep = (r - r.min()) / span * 2 * lam if span > 0 else np.zeros_like(r)
    keep = keep - keep.mean() + (1 - target)
    s = 1 - keep
    assert s.min() >= lo - 1e-9 and s.max() <= hi + 1e-9, (s.min(), s.max())
    return s


def zero_counts(units, target_total):
    """units: list of (numel, sparsity) -> integer zero counts summing exactly to target_total."""
    z = np.array([int(round(n * s)) for n, s in units])
    diff = int(target_total - z.sum())
    i = 0
    while diff != 0:                     # spread the rounding remainder one weight at a time
        step = 1 if diff > 0 else -1
        z[i % len(z)] += step
        diff -= step
        i += 1
    return z


def row_counts(n_zero, out_rows):
    """Distribute n_zero pruned weights over rows: floor for every row, +1 for the first remainder rows."""
    base, rem = divmod(int(n_zero), out_rows)
    k = torch.full((out_rows,), base, dtype=torch.long)
    k[:rem] += 1
    return k


def row_prune_mask_counts(score, k_rows):
    """Per-row pruning with a possibly different count per row (same ranking rule as P2)."""
    din = score.shape[1]
    idx = torch.sort(score, dim=1, stable=True).indices
    prune_pos = torch.arange(din, device=score.device)[None, :] < k_rows.to(score.device)[:, None]
    keep = torch.ones_like(score, dtype=torch.bool)
    keep.scatter_(1, idx, ~prune_pos)
    return keep


def knapsack_bits(units, budget):
    """units: list of dicts {key, options: {bits: (bytes, cost)}}. Minimise total cost subject to
    total bytes <= budget. Start every unit at its cheapest option; repeatedly take the upgrade with
    the largest cost reduction per extra byte (lower convex hull), then fill any remaining bytes."""
    cur = {}
    for u in units:
        b0 = min(u["options"], key=lambda b: u["options"][b][0])
        cur[u["key"]] = b0
    used = sum(u["options"][cur[u["key"]]][0] for u in units)
    if used > budget:
        raise ValueError("budget below the cheapest allocation")
    while True:
        best = None
        for u in units:
            ob, (byt, cst) = cur[u["key"]], u["options"][cur[u["key"]]]
            for b, (byt2, cst2) in u["options"].items():
                if byt2 <= byt or used - byt + byt2 > budget:
                    continue
                gain = (cst - cst2) / (byt2 - byt)
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, u["key"], b, byt2 - byt)
        if best is None:
            break
        _, k, b, dbyt = best
        cur[k] = b
        used += dbyt
    return cur, used
