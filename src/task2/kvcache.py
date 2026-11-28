"""Task 2 Part C: KV-cache editing for Qwen2.5-1.5B-Instruct (FP16, SDPA).

Cache tensors per layer: keys / values [B, Hkv=2, L, dh=128], keys already rotary-encoded at their
original absolute positions. Every policy edits only the pre-question *context* cache; the question
suffix and generated tokens are then appended with their original absolute position IDs
(position_ids = L_ctx + i) and physical cache positions (cache_position = P + i), so retained tokens
keep their RoPE phase and the physical cache stays consistent.

Quantization (all layouts): groups of 64 elements, asymmetric min/max integer grid,
q = clamp(round((x - offset) / scale), 0, 2^b - 1), x_hat = q * scale + offset, scale and offset stored
as FP16 (4 metadata bytes per group), round-half-to-even. Only complete 64-token blocks of the
context are quantized; the trailing L mod 64 tokens stay FP16, and appended tokens stay FP16 until
they complete a 64-token block, which is then quantized (same rule for both layouts). Quantized
values are dequantized back into FP16 cache tensors for ordinary SDPA: packed bytes are a storage
count, GPU memory is that of an FP16 cache.
"""
import math

import torch

G = 64


# ----------------------------------------------------------------------------- quantizer
def _q(x, bits):
    """x [..., 64] -> dequantized x (same dtype). Statistics in FP32, metadata rounded to FP16."""
    xf = x.float()
    mn, mx = xf.amin(-1, keepdim=True), xf.amax(-1, keepdim=True)
    qmax = 2 ** bits - 1
    scale = ((mx - mn) / qmax).clamp(min=1e-6).half().float()
    off = mn.half().float()
    q = torch.clamp(torch.round((xf - off) / scale), 0, qmax)
    return (q * scale + off).to(x.dtype)


def quant_channelwise(x, bits):
    """x [B, H, n*64, d]: groups of 64 consecutive tokens per channel."""
    B, H, T, d = x.shape
    assert T % G == 0
    g = x.reshape(B, H, T // G, G, d).transpose(-1, -2)          # [B,H,n,d,64]
    return _q(g, bits).transpose(-1, -2).reshape(B, H, T, d)


def quant_tokenwise(x, bits):
    """x [B, H, T, d]: groups of 64 consecutive features per token."""
    B, H, T, d = x.shape
    return _q(x.reshape(B, H, T, d // G, G), bits).reshape(B, H, T, d)


LAYOUTS = {"tokenwise": (quant_tokenwise, quant_tokenwise),          # (K, V)
           "kchannel_vtoken": (quant_channelwise, quant_tokenwise)}


# ----------------------------------------------------------------------------- byte accounting
def n_channels(cfg):
    """2 * n_layers * Hkv * dh stored K+V channels (per token: one value per channel)."""
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads)


def fp16_pos_bytes(cfg):
    return 2 * n_channels(cfg)


def quant_bytes(cfg, L, bits):
    """M_b(L) from the guide: complete 64-token groups at b bits + 4 metadata bytes; FP16 residual."""
    return n_channels(cfg) * ((L // G) * (G * bits // 8 + 4) + 2 * (L % G))


def eviction_budget_positions(cfg, budget, index_bytes_per_pos, fixed_index_bytes=0):
    per = fp16_pos_bytes(cfg) + index_bytes_per_pos
    return int((budget - fixed_index_bytes) // per)


def hybrid_bytes(cfg, L, n_fp16_selected_total, bits=2):
    """n_fp16_selected_total = sink + recent + extra; returns (bytes, n_low_quantized, n_fp16_total)."""
    low = L - n_fp16_selected_total
    n_q = (low // G) * G
    n_fp16 = L - n_q
    byt = n_fp16 * fp16_pos_bytes(cfg) + n_channels(cfg) * (n_q // G) * (G * bits // 8 + 4) + math.ceil(L / 8)
    return byt, n_q, n_fp16


# ----------------------------------------------------------------------------- cache helpers
def clone_cache(cache):
    from transformers.cache_utils import DynamicCache
    c = DynamicCache()
    for i, lay in enumerate(cache.layers):
        c.update(lay.keys.clone(), lay.values.clone(), i)
    return c


def select_positions(cache, idx):
    for lay in cache.layers:
        lay.keys = lay.keys[:, :, idx].contiguous()
        lay.values = lay.values[:, :, idx].contiguous()


def quantize_context(cache, L, layout, bits=4):
    """Quantize complete 64-token blocks of the first L positions in place; returns (relK, relV)."""
    fk, fv = LAYOUTS[layout]
    n = (L // G) * G
    ek = ev = nk = nv = 0.0
    for lay in cache.layers:
        K, V = lay.keys[:, :, :n], lay.values[:, :, :n]
        Kq, Vq = fk(K, bits), fv(V, bits)
        ek += float((Kq.float() - K.float()).pow(2).sum()); nk += float(K.float().pow(2).sum())
        ev += float((Vq.float() - V.float()).pow(2).sum()); nv += float(V.float().pow(2).sum())
        lay.keys[:, :, :n] = Kq
        lay.values[:, :, :n] = Vq
    return math.sqrt(ek / nk), math.sqrt(ev / nv)


def quantize_positions(cache, pos, bits=2, layout="kchannel_vtoken"):
    """Quantize the given positions (count a multiple of 64; consecutive groups of 64 in index order)."""
    fk, fv = LAYOUTS[layout]
    ek = ev = nk = nv = 0.0
    for lay in cache.layers:
        K, V = lay.keys[:, :, pos], lay.values[:, :, pos]
        Kq, Vq = fk(K, bits), fv(V, bits)
        ek += float((Kq.float() - K.float()).pow(2).sum()); nk += float(K.float().pow(2).sum())
        ev += float((Vq.float() - V.float()).pow(2).sum()); nv += float(V.float().pow(2).sum())
        lay.keys[:, :, pos] = Kq
        lay.values[:, :, pos] = Vq
    return math.sqrt(ek / nk), math.sqrt(ev / nv)


class BlockQuantizer:
    """Applies the 64-token block rule to entries appended after compression (quantized layouts)."""

    def __init__(self, layout, bits, start):
        self.layout, self.bits, self.start = layout, bits, start   # start = first non-quantized physical pos

    def after_append(self, cache):
        P = cache.layers[0].keys.shape[2]
        fk, fv = LAYOUTS[self.layout]
        while P - self.start >= G:
            s = slice(self.start, self.start + G)
            for lay in cache.layers:
                lay.keys[:, :, s] = fk(lay.keys[:, :, s], self.bits)
                lay.values[:, :, s] = fv(lay.values[:, :, s], self.bits)
            self.start += G


# ----------------------------------------------------------------------------- model steps
@torch.no_grad()
def forward_with_cache(model, ids, cache, abs_pos0, quantizer=None):
    """Append `ids` [1, n] at absolute positions abs_pos0.. and physical positions P..; returns logits."""
    n = ids.shape[1]
    P = cache.get_seq_length()
    dev = ids.device
    out = model(input_ids=ids, past_key_values=cache, use_cache=True,
                position_ids=torch.arange(abs_pos0, abs_pos0 + n, device=dev)[None],
                cache_position=torch.arange(P, P + n, device=dev))
    if quantizer is not None:
        quantizer.after_append(cache)
    return out.logits


@torch.no_grad()
def greedy_decode(model, cache, last_logits, abs_pos0, max_new, eos_ids, quantizer=None):
    toks = []
    logits = last_logits
    for i in range(max_new):
        nxt = int(logits[0, -1].argmax())
        toks.append(nxt)
        if nxt in eos_ids:
            break
        logits = forward_with_cache(model, torch.tensor([[nxt]], device=last_logits.device), cache,
                                    abs_pos0 + i, quantizer)
    return toks


@torch.no_grad()
def question_scores(model, cache, suffix_ids, L):
    """Question-aware importance of the L context positions: FP32 softmax attention of every suffix
    query (all layers, all query heads, causal within the suffix) against the rotary-positioned cached
    keys, summed over layers, heads and suffix tokens. Runs the suffix on a *copy* of the full cache;
    retention is shared by all layers and heads (one global token set)."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    c = clone_cache(cache)
    qs = {}
    hooks = []
    for li, layer in enumerate(model.model.layers):
        def pre(mod, args, kwargs, li=li):
            h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            cos, sin = kwargs["position_embeddings"]
            q = mod.q_proj(h).view(*h.shape[:-1], -1, mod.head_dim).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, q, cos, sin)
            qs[li] = q.float()
        hooks.append(layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        forward_with_cache(model, suffix_ids, c, L)
    finally:
        for h in hooks:
            h.remove()
    n = suffix_ids.shape[1]
    score = torch.zeros(L, device=suffix_ids.device)
    for li, lay in enumerate(c.layers):
        K = lay.keys.float()                                  # [1, Hkv, L+n, d]
        q = qs[li]                                            # [1, Hq, n, d]
        rep = q.shape[1] // K.shape[1]
        K = K.repeat_interleave(rep, dim=1)
        s = (q @ K.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        mask = torch.ones(n, L + n, dtype=torch.bool, device=s.device)
        mask[:, L:] = torch.ones(n, n, dtype=torch.bool, device=s.device).tril()
        s = s.masked_fill(~mask, float("-inf"))
        p = torch.softmax(s, dim=-1)
        score += p[..., :L].sum(dim=(0, 1, 2))
    del c
    return score
