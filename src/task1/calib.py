"""Sequential, block-by-block calibration pass shared by Wanda, AWQ and the sensitivity probes.

The 128 calibration sequences are pushed through the model one Transformer block at a time.
For each block we:
  1. run the block on its current inputs and record, for every distinct projection input,
     the second-moment matrix G = sum_t x_t x_t^T (fp32), the per-channel mean |x| and the token count;
  2. hand those statistics to a method callback that may change the block's eligible weights;
  3. measure each changed projection's relative output error on the same inputs,
        err = ||(W' - W) X||_F^2 / ||W X||_F^2 = tr(dW G dW^T) / tr(W G W^T);
  4. recompute the block outputs with the changed weights, which become the next block's inputs.
So later blocks are calibrated on the outputs of the already-compressed earlier blocks, as in
the reference Wanda/AWQ implementations.
"""
import time

import torch

from src.task1.model_utils import ELIGIBLE

# projections that read the same input share one set of statistics
GROUPS = {"attn_in": ("q_proj", "k_proj", "v_proj"), "o_in": ("o_proj",),
          "mlp_in": ("gate_proj", "up_proj"), "down_in": ("down_proj",)}
GROUP_OF = {p: g for g, ps in GROUPS.items() for p in ps}


def block_linears(layer):
    return {"q_proj": layer.self_attn.q_proj, "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj, "o_proj": layer.self_attn.o_proj,
            "gate_proj": layer.mlp.gate_proj, "up_proj": layer.mlp.up_proj, "down_proj": layer.mlp.down_proj}


class _Stop(Exception):
    pass


@torch.no_grad()
def capture_block0_inputs(model, calib_ids, bs):
    """Hidden states entering block 0 (one tensor per batch) and the shared keyword arguments."""
    dev = next(model.parameters()).device
    layer0 = model.model.layers[0]
    store = {"h": [], "kw": None}

    def pre(mod, args, kwargs):
        store["h"].append(args[0].detach())
        if store["kw"] is None:
            store["kw"] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        raise _Stop

    hdl = layer0.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        for i in range(0, len(calib_ids), bs):
            try:
                model(input_ids=calib_ids[i:i + bs].to(dev), use_cache=False)
            except _Stop:
                pass
    finally:
        hdl.remove()
    kw = store["kw"]
    kw["use_cache"] = False
    kw["past_key_values"] = None
    return store["h"], kw


def _run_block(layer, inps, kw):
    outs = []
    for h in inps:
        o = layer(h, **kw)
        outs.append(o[0] if isinstance(o, tuple) else o)
    return outs


@torch.no_grad()
def collect_stats(layer, inps, kw, need_gram=True):
    lin = block_linears(layer)
    st = {g: {"n": 0, "abs_sum": None, "sq_sum": None, "gram": None} for g in GROUPS}
    hooks = []

    def make(g):
        def hook(mod, args, out):
            x = args[0].reshape(-1, args[0].shape[-1]).float()
            s = st[g]
            s["n"] += x.shape[0]
            a, q = x.abs().sum(0), (x * x).sum(0)
            s["abs_sum"] = a if s["abs_sum"] is None else s["abs_sum"] + a
            s["sq_sum"] = q if s["sq_sum"] is None else s["sq_sum"] + q
            if need_gram:
                G = x.T @ x
                s["gram"] = G if s["gram"] is None else s["gram"] + G
        return hook

    for g, ps in GROUPS.items():
        hooks.append(lin[ps[0]].register_forward_hook(make(g)))
    try:
        _run_block(layer, inps, kw)
    finally:
        for h in hooks:
            h.remove()
    for s in st.values():
        s["act_norm"] = s["sq_sum"].sqrt()          # ||X_j||_2 over all calibration tokens (Wanda)
        s["act_mean_abs"] = s["abs_sum"] / s["n"]   # mean |x_j| (AWQ)
    return st


def rel_output_error(W_old, W_new, gram):
    dW = (W_new.float() - W_old.float())
    W = W_old.float()
    num = ((dW @ gram) * dW).sum()
    den = ((W @ gram) * W).sum()
    return float(num / den)


@torch.no_grad()
def sequential_pass(model, calib_ids, method, bs=8, need_gram=True, log_every=4):
    """method(layer_idx, linears: dict[name->nn.Linear], stats) changes weights in place and may
    return a dict of extra per-projection records. Returns a list of per-projection records."""
    t0 = time.time()
    inps, kw = capture_block0_inputs(model, calib_ids, bs)
    records = []
    for li, layer in enumerate(model.model.layers):
        st = collect_stats(layer, inps, kw, need_gram=need_gram)
        lin = block_linears(layer)
        old = {p: m.weight.detach().clone() for p, m in lin.items()}
        extra = method(li, lin, st) or {}
        for p in ELIGIBLE:
            g = st[GROUP_OF[p]]
            rec = {"layer": li, "proj": p, "numel": lin[p].weight.numel(),
                   "nonzero": int((lin[p].weight != 0).sum())}
            if need_gram:
                rec["rel_out_err"] = rel_output_error(old[p], lin[p].weight, g["gram"])
            rec.update(extra.get(p, {}))
            records.append(rec)
        del old, st
        inps = _run_block(layer, inps, kw)
        if (li + 1) % log_every == 0 or li == len(model.model.layers) - 1:
            print(f"  block {li + 1}/{len(model.model.layers)} done ({time.time() - t0:.0f}s)", flush=True)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return records, time.time() - t0
