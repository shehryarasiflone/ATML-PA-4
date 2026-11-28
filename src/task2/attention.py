"""Task 2 attention operators.

Part A (B, H, L, d) reference implementations:
  dense_causal  explicit scores, causal mask, max-subtracted softmax, multiply by V (in the input dtype)
  tiled_causal  exact blockwise forward with online softmax (FP32 running max / normaliser / accumulator);
                never materialises the L x L matrix in the forward pass. Its backward is plain autograd
                through the Python loop, which stores every tile's intermediates (not a FlashAttention
                backward kernel).
  sdpa_causal   torch.nn.functional.scaled_dot_product_attention(is_causal=True)
Part B (MQAR models, B, H, T, d):
  boolean masks for sliding-window and mixed local/dilated heads (explicit dense masks: a correctness
  implementation, not an efficient sparse kernel), and causal ELU+1 linear attention in a masked
  quadratic form (training) and a chunked recurrent form (state S_t, z_t; inference sweep).
"""
import math

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------------- Part A
def dense_causal(q, k, v, allowed=None):
    """allowed: optional bool [L, L] (or broadcastable) mask, True = may attend. Default: causal.
    A fully masked row returns zeros (defined behaviour) instead of NaN."""
    L, S = q.shape[-2], k.shape[-2]
    if allowed is None:
        allowed = torch.ones(L, S, dtype=torch.bool, device=q.device).tril()
    s = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(q.shape[-1]))
    s = s.masked_fill(~allowed, float("-inf"))
    m = s.amax(-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    p = torch.exp(s - m)
    l = p.sum(-1, keepdim=True)
    out = p @ v
    return torch.where(l > 0, out / l.clamp(min=torch.finfo(l.dtype).tiny), torch.zeros_like(out))


def tiled_causal(q, k, v, tile=128):
    """Exact causal attention with query and key/value tiles and the online-softmax update
    m' = max(m, max s), l' = e^(m-m') l + sum e^(s-m'), o' = e^(m-m') o + sum e^(s-m') V."""
    B, H, L, d = q.shape
    scale = 1.0 / math.sqrt(d)
    outs = []
    for i0 in range(0, L, tile):
        i1 = min(i0 + tile, L)
        qi = q[:, :, i0:i1].float() * scale
        m = torch.full((B, H, i1 - i0, 1), float("-inf"), device=q.device)
        l = torch.zeros((B, H, i1 - i0, 1), device=q.device)
        o = torch.zeros((B, H, i1 - i0, d), device=q.device)
        for j0 in range(0, i1, tile):           # key tiles strictly in the future are skipped
            j1 = min(j0 + tile, L)
            s = qi @ k[:, :, j0:j1].float().transpose(-1, -2)
            if j1 > i0:                         # diagonal tile: apply the causal mask
                qpos = torch.arange(i0, i1, device=q.device)[:, None]
                kpos = torch.arange(j0, j1, device=q.device)[None, :]
                s = s.masked_fill(kpos > qpos, float("-inf"))
            m_new = torch.maximum(m, s.amax(-1, keepdim=True))
            alpha = torch.exp(m - m_new)
            p = torch.exp(s - m_new)
            l = alpha * l + p.sum(-1, keepdim=True)
            o = alpha * o + p @ v[:, :, j0:j1].float()
            m = m_new
        outs.append((o / l).to(q.dtype))
    return torch.cat(outs, dim=2)


def sdpa_causal(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


# ------------------------------------------------------------------------- Part B masks
def window_mask(T, W, device):
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    return (j <= i) & (i - j < W)


def dilated_head_mask(T, device, step=4, n=64):
    """Offsets {0, 4, ..., 252}: query i may attend j = i - off."""
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    off = i - j
    return (off >= 0) & (off % step == 0) & (off <= step * (n - 1))


def mixed_dilated_mask(T, device, W=64):
    """[H=4, T, T]: heads 0-1 local window W, heads 2-3 dilated offsets."""
    loc, dil = window_mask(T, W, device), dilated_head_mask(T, device)
    return torch.stack([loc, loc, dil, dil])


def elu1(x):
    return F.elu(x) + 1.0


def linear_attention_quadratic(q, k, v, eps=1e-6):
    """Causal ELU+1 linear attention written as a masked T x T product (exactly equal to the
    recurrence; materialises a T x T matrix, which is cheap at T = 256 and is reported as such)."""
    fq, fk = elu1(q.float()), elu1(k.float())
    A = (fq @ fk.transpose(-1, -2)).tril()
    return ((A @ v.float()) / (A.sum(-1, keepdim=True) + eps)).to(q.dtype)


def linear_attention_recurrent(q, k, v, eps=1e-6, chunk=64):
    """Chunked causal recurrence with persistent state S (d_k x d_v) and z (d_k) per head.
    Inside a chunk the contribution of earlier chunk tokens comes from the carried state and the
    within-chunk part is a small masked product, so memory is O(T * chunk) not O(T^2)."""
    B, H, T, d = q.shape
    fq, fk, vf = elu1(q.float()), elu1(k.float()), v.float()
    S = torch.zeros(B, H, d, vf.shape[-1], device=q.device)
    z = torch.zeros(B, H, d, device=q.device)
    outs = []
    for c0 in range(0, T, chunk):
        c1 = min(c0 + chunk, T)
        qc, kc, vc = fq[:, :, c0:c1], fk[:, :, c0:c1], vf[:, :, c0:c1]
        A = (qc @ kc.transpose(-1, -2)).tril()
        num = qc @ S + A @ vc
        den = (qc * z[:, :, None, :]).sum(-1, keepdim=True) + A.sum(-1, keepdim=True)
        outs.append(num / (den + eps))
        S = S + kc.transpose(-1, -2) @ vc
        z = z + kc.sum(2)
    return torch.cat(outs, dim=2).to(q.dtype)
