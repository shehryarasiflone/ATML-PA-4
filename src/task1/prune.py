"""Pruning masks: per-output-row unstructured (P1 magnitude, P2 Wanda, P3 nonuniform Wanda) and 2:4."""
import torch


def magnitude_score(W, st=None):
    return W.float().abs()


def wanda_score(W, st):
    """|W_ij| * ||X_j||_2 (Sun et al., 2024)."""
    return W.float().abs() * st["act_norm"].to(W.device)[None, :]


def row_prune_mask(score, sparsity):
    """Boolean keep-mask zeroing the floor(s * d_in) lowest-score weights in every output row.
    Ties are broken by column index (stable sort) so masks are deterministic."""
    d_in = score.shape[1]
    k = int(sparsity * d_in)
    keep = torch.ones_like(score, dtype=torch.bool)
    if k > 0:
        idx = torch.sort(score, dim=1, stable=True).indices[:, :k]
        keep.scatter_(1, idx, False)
    return keep


def nm_prune_mask(score, n=2, m=4):
    """Keep the n highest-score weights in each group of m contiguous input columns."""
    out, d_in = score.shape
    assert d_in % m == 0
    s = score.reshape(out, d_in // m, m)
    idx = torch.topk(s, n, dim=-1).indices
    keep = torch.zeros_like(s, dtype=torch.bool).scatter_(-1, idx, True)
    return keep.reshape(out, d_in)


def mask_overlap(keep_a, keep_b):
    """Share of the weights pruned by A that B also prunes (1 = identical pruned sets)."""
    pa, pb = ~keep_a, ~keep_b
    n = int(pa.sum())
    return float((pa & pb).sum()) / n if n else 1.0
