"""Task 1 model helpers: loading, eligible projections, parameter counts."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-1.5B"
BASE_REV = "8faed761d45a263340a0528343f099c05c9a4323"   # recorded by env_check.py
ELIGIBLE = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_tokenizer():
    return AutoTokenizer.from_pretrained(BASE, revision=BASE_REV)


def load_base(device="cuda", dtype=torch.float16):
    """A fresh, unchanged FP16 copy. Every compression condition starts from one of these."""
    m = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REV, dtype=dtype,
                                             attn_implementation="sdpa")
    return m.to(device).eval()


def eligible_linears(model):
    """[(name, nn.Linear)] for the 7 projections in every block, in layer order."""
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in ELIGIBLE and ".layers." in name:
            out.append((name, mod))
    return out


def layer_and_type(name):
    """'model.layers.13.mlp.down_proj' -> (13, 'down_proj')"""
    parts = name.split(".")
    return int(parts[parts.index("layers") + 1]), parts[-1]


def unique_params(model):
    """Parameters with tied tensors counted once: [(name, tensor)]."""
    seen, out = set(), []
    for n, p in model.named_parameters():
        if p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        out.append((n, p))
    return out


def param_accounting(model):
    """Eligible vs. non-eligible parameter counts (tied weights once)."""
    elig = eligible_linears(model)
    elig_ptrs = {m.weight.data_ptr() for _, m in elig}
    n_elig = sum(m.weight.numel() for _, m in elig)
    n_other = sum(p.numel() for _, p in unique_params(model) if p.data_ptr() not in elig_ptrs)
    nonzero_elig = sum(int((m.weight != 0).sum()) for _, m in elig)
    return {"eligible_matrices": len(elig), "eligible_weights": n_elig,
            "eligible_nonzero": nonzero_elig, "eligible_nonzero_fraction": nonzero_elig / n_elig,
            "other_params_uncompressed": n_other, "total_params_tied_once": n_elig + n_other}
