"""Task 1, steps 5-6: Q1 uniform RTN and Q2 AWQ-style quantization at 8/4/3 bits, group size 128.

  python scripts/t1_quant.py --methods rtn awq --bits 8 4 3
Every condition starts from a fresh FP16 checkpoint; finished conditions are skipped.
Also writes results/task1/q2_fold_check.json: the AWQ rescaling alone (no rounding) must leave
the model's outputs unchanged up to FP16 rounding.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.common.utils import SEED, save_json, set_seed
from src.task1.awq import ALPHAS, awq_method
from src.task1.calib import sequential_pass
from src.task1.evaluate import T1_DIR, load_calibration, load_dev_windows, load_hellaswag, perplexity
from src.task1.model_utils import load_base, load_tokenizer
from src.task1.quant import GROUP, fake_quant, model_packed_bytes
from src.task1.runner import done, finish, other_params, shapes

p = argparse.ArgumentParser()
p.add_argument("--methods", nargs="+", default=["rtn", "awq"], choices=["rtn", "awq"])
p.add_argument("--bits", nargs="+", type=int, default=[8, 4, 3])
p.add_argument("--bs", type=int, default=8)
args = p.parse_args()
set_seed(SEED)
tok = load_tokenizer()
calib, _ = load_calibration("calibration")
hs_all = load_hellaswag(tok)

QCONV = {"group_size": GROUP, "symmetric": False, "rounding": "round-half-to-even", "scale": "FP16 per group",
         "zero_point": "b-bit unsigned per group", "clipping": "none (min/max)", "padding": "none needed",
         "packing": "contiguous b-bit codes per matrix, rounded up to bytes"}


@torch.no_grad()
def fold_check():
    f = T1_DIR / "q2_fold_check.json"
    if f.exists():
        return
    print("\n===== AWQ fold check (scaling only, no rounding) =====", flush=True)
    base = load_base()
    scaled = load_base()
    method, _ = awq_method(lambda li, p: 4)
    sequential_pass(scaled, calib, method, bs=args.bs, post=None)
    ids = torch.tensor([w["token_ids"] for w in load_dev_windows()[:4]]).cuda()
    a, b = base(input_ids=ids).logits.float(), scaled(input_ids=ids).logits.float()
    pa, pb = perplexity(base, load_dev_windows())[0], perplexity(scaled, load_dev_windows())[0]
    out = {"max_abs_logit_diff": float((a - b).abs().max()), "mean_abs_logit_diff": float((a - b).abs().mean()),
           "top1_agreement": float((a.argmax(-1) == b.argmax(-1)).float().mean()),
           "dev_ppl_base": pa["ppl"], "dev_ppl_scaled_unrounded": pb["ppl"]}
    save_json(out, f)
    print("fold check:", out, flush=True)
    del base, scaled
    torch.cuda.empty_cache()


for method in args.methods:
    if method == "awq":
        fold_check()
    for b in args.bits:
        tag = f"{'Q1_rtn' if method == 'rtn' else 'Q2_awq'}_b{b}"
        if done(tag):
            print(f"[{tag}] already done, skipping", flush=True)
            continue
        print(f"\n===== {tag} =====", flush=True)
        model = load_base()
        torch.cuda.reset_peak_memory_stats()
        if method == "rtn":
            def m_noop(li, lin, st, ctx):
                return None

            def post(li, lin, ctx, b=b):
                for mod in lin.values():
                    mod.weight.copy_(fake_quant(mod.weight, b).to(mod.weight.dtype))
            recs, sec = sequential_pass(model, calib, m_noop, bs=args.bs, post=post)
            extra_fp16 = 0
        else:
            m_awq, post = awq_method(lambda li, p, b=b: b)
            recs, sec = sequential_pass(model, calib, m_awq, bs=args.bs, post=post)
            extra_fp16 = sum(s[1] for (lt, s) in shapes(model) if lt[1] == "o_proj")   # o_proj input scales
        peak = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        stor = model_packed_bytes([(o, i, b) for _, (o, i) in shapes(model)], other_params(model), extra_fp16)
        stor["nominal_bits_per_eligible_weight"] = b
        stor["effective_bits_per_eligible_weight"] = 8 * stor["eligible_packed_bytes"] / sum(
            o * i for _, (o, i) in shapes(model))
        meta = {"method": method, "bits": b, "quantizer": QCONV, "execution": "dequantized FP16 weights, dense kernels"}
        if method == "awq":
            meta["alpha_grid"] = ALPHAS
        finish(model, tok, tag, recs, sec, peak, stor, meta, "full" if b in (3, 4) else "diag", hs_all)
print("\nAll requested quantization conditions done.", flush=True)
