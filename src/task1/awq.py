"""AWQ-style activation-aware scaling (Lin et al., 2024), reproduced parts and simplifications:

Reproduced: per-input-channel scales s = mean|x|^alpha normalised by sqrt(max(s) * min(s)); a grid
search over alpha on calibration activations; scales chosen per group of projections that share an
input; inverse scales folded into the preceding operation when algebraically valid
(input_layernorm -> q/k/v, post_attention_layernorm -> gate/up, up_proj rows -> down_proj);
block-by-block processing on full-precision activations; the same quantizer as RTN afterwards.
Simplified / not reproduced: the search loss is the summed output MSE of the group's linear
projections (computed exactly from X^T X) rather than the output of the whole attention/MLP module;
an 11-point alpha grid {0, 0.1, ..., 1.0} instead of 20 points; no weight-clipping search; no fused
low-bit kernels. o_proj (GQA: 12 query heads share 2 value heads, so v_proj cannot absorb a free
per-channel scale) keeps its own scales, serialized as FP16 and counted in storage.
alpha = 0 gives s = 1, i.e. plain RTN, so the search can never be worse than Q1 on its own criterion.
"""
import torch

from src.task1.calib import GROUPS
from src.task1.quant import fake_quant

ALPHAS = [round(0.1 * i, 1) for i in range(11)]


def _scale(act_mean_abs, alpha):
    s = act_mean_abs.clamp(min=1e-4).pow(alpha)
    s = s / (s.max() * s.min()).sqrt()
    return s.clamp(min=1e-4)


def _group_loss(Ws, bits_list, s, gram):
    loss = 0.0
    for W, b in zip(Ws, bits_list):
        Wf = W.float()
        Weff = fake_quant(Wf * s[None, :], b) / s[None, :]
        dW = Weff - Wf
        loss += float(((dW @ gram) * dW).sum())
    return loss


def search_group(Ws, bits_list, st):
    """Return best alpha, its scales, and the loss of every grid point (for the record)."""
    grid = {}
    best = None
    for a in ALPHAS:
        s = _scale(st["act_mean_abs"], a)
        loss = _group_loss(Ws, bits_list, s, st["gram"])
        grid[a] = loss
        if best is None or loss < best[1]:
            best = (a, loss, s)
    return best[0], best[2], grid


def group_rel_errors(Ws, bits_list, s, gram):
    """Relative output error of each projection after scaled quantization (original coordinates)."""
    out = []
    for W, b in zip(Ws, bits_list):
        Wf = W.float()
        dW = fake_quant(Wf * s[None, :], b) / s[None, :] - Wf
        out.append(float(((dW @ gram) * dW).sum() / ((Wf @ gram) * Wf).sum()))
    return out


@torch.no_grad()
def awq_method(bits_of):
    """Build (method, post) callbacks for sequential_pass. bits_of(layer_idx, proj) -> bits."""
    state = {}

    def method(li, lin, st, ctx):
        layer = ctx["layer"]
        rec = {}
        h0, kw = ctx["inps"][0], ctx["kw"]
        ref = layer(h0, **kw)
        ref = ref[0] if isinstance(ref, tuple) else ref
        scales = {}
        for g, projs in GROUPS.items():
            Ws = [lin[p].weight for p in projs]
            bl = [bits_of(li, p) for p in projs]
            a, s, grid = search_group(Ws, bl, st[g])
            errs = group_rel_errors(Ws, bl, s, st[g]["gram"])
            rtn = group_rel_errors(Ws, bl, torch.ones_like(s), st[g]["gram"])
            for p, b, e, r in zip(projs, bl, errs, rtn):
                rec[p] = {"bits": b, "alpha": a, "rel_out_err": e, "rtn_rel_out_err": r,
                          "act_max_over_median": float(st[g]["act_mean_abs"].max() / st[g]["act_mean_abs"].median())}
            scales[g] = s
        # fold the full-precision rescaling (function preserving before rounding)
        sa = scales["attn_in"].to(layer.input_layernorm.weight.dtype)
        layer.input_layernorm.weight.div_(sa)
        for p in GROUPS["attn_in"]:
            lin[p].weight.mul_(sa[None, :])
        sm = scales["mlp_in"].to(layer.post_attention_layernorm.weight.dtype)
        layer.post_attention_layernorm.weight.div_(sm)
        for p in GROUPS["mlp_in"]:
            lin[p].weight.mul_(sm[None, :])
        sd = scales["down_in"].to(lin["up_proj"].weight.dtype)
        lin["up_proj"].weight.div_(sd[:, None])
        lin["down_proj"].weight.mul_(sd[None, :])
        state[li] = scales["o_in"]
        new = layer(h0, **kw)
        new = new[0] if isinstance(new, tuple) else new
        rec["q_proj"]["fold_check_max_abs_diff"] = float((new.float() - ref.float()).abs().max())
        rec["q_proj"]["fold_check_ref_max_abs"] = float(ref.float().abs().max())
        return rec

    def post(li, lin, ctx):
        for p, m in lin.items():
            b = bits_of(li, p)
            W = m.weight
            if p == "o_proj":
                s = state[li].to(W.device)
                W.copy_((fake_quant(W.float() * s[None, :], b) / s[None, :]).to(W.dtype))
            else:
                W.copy_(fake_quant(W, b).to(W.dtype))
        return {}

    return method, post
