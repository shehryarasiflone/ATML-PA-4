"""Task 2 Part B: MQAR data, the small Transformer, training and evaluation.

Model (identical for every attention method): learned token embeddings (vocab 256) + fixed sinusoidal
positions, 2 pre-norm blocks (LayerNorm -> attention -> residual; LayerNorm -> 4x GELU MLP -> residual),
hidden 128, 4 heads x 32, final LayerNorm, linear classifier over the 64 value IDs (128-191).
Loss: cross-entropy at the K marked query positions only (query-position classification).
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.data import data_root, records
from src.task2.attention import linear_attention_quadratic, mixed_dilated_mask, window_mask

T = 256
VAL_LO = 128
N_CLASSES = 64
BINS = [(0, 32), (32, 64), (64, 128), (128, 256)]


# ------------------------------------------------------------------------------ data
def generator_module():
    """The supplied generator (scripts/mqar_generator.py of the course data repo)."""
    gdir = data_root().parent / "scripts"
    if str(gdir) not in sys.path:
        sys.path.insert(0, str(gdir))
    import mqar_generator
    return mqar_generator


def to_arrays(exs, kmax=16):
    n = len(exs)
    tok = np.zeros((n, T), dtype=np.int16)
    qpos = np.full((n, kmax), -1, dtype=np.int16)
    ans = np.zeros((n, kmax), dtype=np.int16)
    dist = np.zeros((n, kmax), dtype=np.int16)
    for i, e in enumerate(exs):
        tok[i] = e["tokens"]
        K = len(e["query_positions"])
        qpos[i, :K] = e["query_positions"]
        ans[i, :K] = e["answer_ids"]
        dist[i, :K] = e["distances"]
    return {"tokens": tok, "qpos": qpos, "ans": ans, "dist": dist}


def load_eval(split):
    """split: 'test' (9,216) or 'development' (2,304) -> {(K, regime): arrays}"""
    rows = records(data_root() / "mqar" / f"mqar_{split}.jsonl.gz")
    out = {}
    for K in (4, 8, 16):
        for reg in ("near", "far", "mixed"):
            sel = [r for r in rows if r["K"] == K and r["regime"] == reg]
            out[(K, reg)] = to_arrays(sel)
    return out


def train_index(seed, step, b, bs=32):
    """Disjoint deterministic training stream per seed (the generator's 'train' split)."""
    return (seed - 6304) * 10_000_000 + step * bs + b


# ------------------------------------------------------------------------------ model
def sinusoidal(T, d):
    pos = torch.arange(T)[:, None].float()
    i = torch.arange(0, d, 2).float()
    ang = pos / (10000 ** (i / d))
    pe = torch.zeros(T, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe


class Attention(nn.Module):
    def __init__(self, d=128, h=4, kind="dense", W=64):
        super().__init__()
        self.h, self.dh, self.kind, self.W = h, d // h, kind, W
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, kind=None):
        kind = kind or self.kind
        B, Tn, d = x.shape
        q, k, v = self.qkv(x).view(B, Tn, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if kind == "dense":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif kind == "window":
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=window_mask(Tn, self.W, x.device))
        elif kind == "dilated":
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mixed_dilated_mask(Tn, x.device, self.W))
        elif kind == "linear":
            o = linear_attention_quadratic(q, k, v)
        elif kind == "hybrid":            # B3 intervention: 2 linear (global) heads + 2 window heads
            ol = linear_attention_quadratic(q[:, :2], k[:, :2], v[:, :2])
            ow = F.scaled_dot_product_attention(q[:, 2:], k[:, 2:], v[:, 2:],
                                                attn_mask=window_mask(Tn, self.hybrid_W, x.device))
            o = torch.cat([ol, ow], dim=1)
        else:
            raise ValueError(kind)
        return self.out(o.transpose(1, 2).reshape(B, Tn, d))


class Block(nn.Module):
    def __init__(self, d, h, kind, W):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, h, kind, W)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, kind=None):
        x = x + self.attn(self.ln1(x), kind)
        return x + self.mlp(self.ln2(x))


class MQARModel(nn.Module):
    def __init__(self, kind="dense", d=128, h=4, n_layers=2, W=64, vocab=256, hybrid_W=None):
        super().__init__()
        self.kind = kind
        self.emb = nn.Embedding(vocab, d)
        self.register_buffer("pe", sinusoidal(T, d), persistent=False)
        self.blocks = nn.ModuleList([Block(d, h, kind, W) for _ in range(n_layers)])
        for b in self.blocks:
            b.attn.hybrid_W = hybrid_W
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_CLASSES)

    def forward(self, tokens, qpos, kind=None):
        x = self.emb(tokens) + self.pe[: tokens.shape[1]]
        for b in self.blocks:
            x = b(x, kind)
        x = self.ln_f(x)
        idx = qpos.clamp(min=0).long()
        h = torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))
        return self.head(h)            # [B, Kmax, 64]; padded slots are masked in the loss


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def batch_to(arr, sl, device):
    t = torch.as_tensor(arr["tokens"][sl].astype(np.int64), device=device)
    q = torch.as_tensor(arr["qpos"][sl].astype(np.int64), device=device)
    a = torch.as_tensor(arr["ans"][sl].astype(np.int64), device=device) - VAL_LO
    d = torch.as_tensor(arr["dist"][sl].astype(np.int64), device=device)
    return t, q, a, d


def loss_acc(model, t, q, a, kind=None):
    logits = model(t, q, kind)
    valid = q >= 0
    loss = F.cross_entropy(logits[valid], a[valid])
    acc = (logits.argmax(-1)[valid] == a[valid]).float().mean()
    return loss, acc, logits, valid


@torch.no_grad()
def evaluate(model, sets, device, kind=None, bs=256, keep_correct=False):
    model.eval()
    res, correct = {}, {}
    for (K, reg), arr in sets.items():
        n = len(arr["tokens"])
        tot_loss, cor, cnt = 0.0, [], []
        dists = []
        for i in range(0, n, bs):
            t, q, a, d = batch_to(arr, slice(i, i + bs), device)
            logits = model(t, q, kind)
            valid = q >= 0
            tot_loss += F.cross_entropy(logits[valid], a[valid], reduction="sum").item()
            cor.append((logits.argmax(-1) == a)[valid].cpu())
            dists.append(d[valid].cpu())
        c, dd = torch.cat(cor).float(), torch.cat(dists)
        r = {"acc": float(c.mean()), "loss": tot_loss / len(c), "n_queries": len(c)}
        for lo, hi in BINS:
            sel = (dd >= lo) & (dd < hi)
            r[f"acc_d{lo}_{hi}"] = float(c[sel].mean()) if sel.any() else None
            r[f"n_d{lo}_{hi}"] = int(sel.sum())
        res[f"K{K}_{reg}"] = r
        if keep_correct:
            correct[f"K{K}_{reg}"] = (c.numpy().astype(np.uint8), dd.numpy().astype(np.int16))
    model.train()
    return (res, correct) if keep_correct else res


def lr_at(step, peak=8e-4, warm=200, total=4500, floor=0.15):
    if step < warm:
        return peak * (step + 1) / warm
    return peak * (1 - (1 - floor) * (step - warm) / (total - warm))


# ------------------------------------------------------------------------------ training streams
CACHE = Path("/kaggle/working/mqar_cache") if Path("/kaggle/working").exists() else Path("mqar_cache")


def _gen_chunk(args):
    K, reg, idxs = args
    g = generator_module()
    return [g.make_example("train", K, reg, i) for i in idxs]


def train_stream(K, reg, seed, steps=4500, bs=32, workers=2):
    """Training examples for one (K, regime, seed), generated once with the supplied generator and cached."""
    import os
    from multiprocessing import Pool
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"train_K{K}_{reg}_s{seed}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    n = steps * bs
    idx = [train_index(seed, 0, b) for b in range(n)]       # = offset + 0..n-1
    chunks = [(K, reg, idx[i:i + 2000]) for i in range(0, n, 2000)]
    with Pool(workers) as p:
        exs = [e for part in p.map(_gen_chunk, chunks) for e in part]
    arr = to_arrays(exs)
    arr["tokens"] = arr["tokens"].astype(np.uint8)
    tmp = f.with_suffix(f".tmp{os.getpid()}.npz")
    np.savez(tmp, **arr)
    os.replace(tmp, f)
    return arr
