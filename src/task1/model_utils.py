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


def eligible_shapes(config=None):
    """[((layer, proj), (out, in))] in the same order as eligible_linears(), from the config only."""
    if config is None:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(BASE, revision=BASE_REV)
    H, I = config.hidden_size, config.intermediate_size
    kv = config.num_key_value_heads * (H // config.num_attention_heads)
    shp = {"q_proj": (H, H), "k_proj": (kv, H), "v_proj": (kv, H), "o_proj": (H, H),
           "gate_proj": (I, H), "up_proj": (I, H), "down_proj": (H, I)}
    return [((li, p), shp[p]) for li in range(config.num_hidden_layers) for p in ELIGIBLE]
