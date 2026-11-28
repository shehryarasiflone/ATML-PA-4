"""Task 2 Part B resources: theoretical pair counts / state bytes at T=256 and a forward-only
inference sweep at T in {256, 1024, 4096, 8192} with fixed random q, k, v (B=4, H=4, d=32).

  python scripts/t2b_costs.py
Implementation notes recorded in the output: window and dilated attention use explicit dense boolean
masks with SDPA (mask storage is reported; this is not an efficient sparse kernel); linear attention
is timed both in the masked quadratic form and in the chunked recurrent form (persistent state).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
import torch.nn.functional as F

from src.common.utils import RESULTS_DIR, SEED, save_json
from src.task2.attention import (linear_attention_quadratic, linear_attention_recurrent, mixed_dilated_mask,
                                 window_mask)

OUT = RESULTS_DIR / "task2" / "mqar"
OUT.mkdir(parents=True, exist_ok=True)
T, H, DH, W = 256, 4, 32, 64


def pairs(mask):
    return int(mask.sum())


dev = "cpu"
dense = torch.ones(T, T, dtype=torch.bool).tril()
win = window_mask(T, W, dev)
dil = mixed_dilated_mask(T, dev, W)
lin_state = DH * DH + DH                                   # S (d x d) + z (d) per head
theory = {
    "T": T, "heads": H, "head_dim": DH,
    "pairs_per_layer": {"dense": H * pairs(dense), "window64": H * pairs(win), "dilated_mixed": pairs(dil),
                        "linear": "no pairs (state recurrence)"},
    "max_positions_per_query": {"dense": T, "window64": W, "dilated_head": 64, "linear": "all (compressed)"},
    "two_layer_receptive_field": {"window64": 2 * (W - 1), "dilated_mixed": "up to 252 at offsets divisible by 4, "
                                  "composed with 63-step local hops"},
    "persistent_state_floats_per_layer_at_decode": {
        "dense_kv_cache": 2 * H * DH * T, "window64_kv_buffer": 2 * H * DH * (W - 1),
        "linear_S_z": H * lin_state, "hybrid_2linear_2window34": 2 * lin_state + 2 * 2 * DH * 33},
    "state_bytes_per_layer": {
        "dense_kv_cache_fp32": 4 * 2 * H * DH * T, "window64_kv_buffer_fp32": 4 * 2 * H * DH * (W - 1),
        "linear_fp32": 4 * H * lin_state,
        "hybrid_linear_fp32_plus_window34_fp16": 4 * 2 * lin_state + 2 * 2 * 2 * DH * 33},
    "hybrid_note": "B3 intervention: 2 heads causal ELU+1 linear (FP32 state) + 2 heads sliding window W=34 whose "
                   "33 past keys/values are stored in FP16; W=34 is the largest window for which per-layer state "
                   "bytes equal the 4-head linear-attention comparator (16,896 bytes).",
}
assert theory["state_bytes_per_layer"]["hybrid_linear_fp32_plus_window34_fp16"] == theory["state_bytes_per_layer"]["linear_fp32"]
print(theory, flush=True)
save_json(theory, OUT / "theoretical_costs.json")

dev = "cuda"
rows = []
B = 4
for L in [256, 1024, 4096, 8192]:
    g = torch.Generator(device=dev).manual_seed(SEED + L)
    q, k, v = (torch.randn(B, H, L, DH, device=dev, generator=g) for _ in range(3))
    impls = {
        "dense_sdpa_causal": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
        "window64_sdpa_densemask": lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=window_mask(L, W, dev)),
        "dilated_sdpa_densemask": lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mixed_dilated_mask(L, dev, W)),
        "linear_quadratic_masked": lambda: linear_attention_quadratic(q, k, v),
        "linear_recurrent_chunk64": lambda: linear_attention_recurrent(q, k, v, chunk=64),
    }
    for name, fn in impls.items():
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            ts = []
            with torch.no_grad():
                for r in range(13):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    fn()
                    torch.cuda.synchronize()
                    if r >= 3:
                        ts.append((time.perf_counter() - t0) * 1000)
            rows.append({"T": L, "impl": name, "status": "ok", "ms_mean": sum(ts) / len(ts), "ms_min": min(ts),
                         "peak_extra_alloc_mb": (torch.cuda.max_memory_allocated() - base) / 2**20})
        except Exception as e:  # noqa: BLE001
            rows.append({"T": L, "impl": name, "status": "out_of_memory" if "out of memory" in str(e).lower()
                         else str(e)[:60]})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "inference_sweep.csv", index=False)
print("Done.", flush=True)
