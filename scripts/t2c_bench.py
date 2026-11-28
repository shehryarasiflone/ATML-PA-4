"""Task 2 C5 systems benchmark: real packed INT4 KV cache (channel-wise K / token-wise V, groups of 64,
FP16 scale + FP16 offset, two 4-bit codes per byte, FP16 trailing L mod 64 tokens).

  python scripts/t2c_bench.py
For B in {1, 8} and L in {2048, 4096} (random cache contents; costs do not depend on values):
pack (quantize + bit-pack) time, unpack (unpack + dequantize to FP16) time, serialized bytes vs the
formula and vs FP16, and one decode step of Qwen2.5-1.5B-Instruct with an FP16 cache vs. with
unpacking first. 3 warm-up + 10 timed CUDA-synchronised repetitions.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from src.common.utils import RESULTS_DIR, SEED
from src.task2.kvcache import quant_bytes

TARGET, REV = "Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
OUT = RESULTS_DIR / "task2" / "kv"
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"
G, QMAX = 64, 15


def _enc(g):
    """g [..., 64] FP16 -> packed uint8 [..., 32], scale fp16, offset fp16."""
    gf = g.float()
    mn, mx = gf.amin(-1, keepdim=True), gf.amax(-1, keepdim=True)
    sc = ((mx - mn) / QMAX).clamp(min=1e-6).half()
    off = mn.half()
    q = torch.clamp(torch.round((gf - off.float()) / sc.float()), 0, QMAX).to(torch.uint8)
    return (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous(), sc.squeeze(-1), off.squeeze(-1)


def _dec(p, sc, off):
    q = torch.stack([p & 15, p >> 4], dim=-1).flatten(-2).half()
    return q * sc[..., None] + off[..., None]


def pack_layer(K, V):
    B, H, L, d = K.shape
    n = (L // G) * G
    kb = K[:, :, :n].reshape(B, H, n // G, G, d).transpose(-1, -2)          # channel-wise K
    vb = V[:, :, :n].reshape(B, H, n, d // G, G)                            # token-wise V
    return {"k": _enc(kb), "v": _enc(vb), "rk": K[:, :, n:].clone(), "rv": V[:, :, n:].clone(), "shape": (B, H, L, d)}


def unpack_layer(pk):
    B, H, L, d = pk["shape"]
    n = (L // G) * G
    K = _dec(*pk["k"]).transpose(-1, -2).reshape(B, H, n, d)
    V = _dec(*pk["v"]).reshape(B, H, n, d)
    return torch.cat([K, pk["rk"]], 2), torch.cat([V, pk["rv"]], 2)


def nbytes(pk):
    return sum(t.nbytes for t in (*pk["k"], *pk["v"], pk["rk"], pk["rv"]))


def timeit(fn, reps=10, warm=3):
    ts = []
    for r in range(warm + reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        if r >= warm:
            ts.append(1000 * (time.perf_counter() - t0))
    return sum(ts) / len(ts), min(ts)


model = AutoModelForCausalLM.from_pretrained(TARGET, revision=REV, dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()
cfg = model.config
nl, hkv, d = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
rows = []
# correctness of pack/unpack against the simulated quantizer used in t2c_kv.py
from src.task2.kvcache import quant_channelwise, quant_tokenwise  # noqa: E402
x = torch.randn(1, hkv, 256, d, device=dev, dtype=torch.float16)
kk, vv = unpack_layer(pack_layer(x, x))
check = {"pack_unpack_vs_simulated_K_maxabs": float((kk - quant_channelwise(x, 4)).abs().max()),
         "pack_unpack_vs_simulated_V_maxabs": float((vv - quant_tokenwise(x, 4)).abs().max())}
print(check, flush=True)
g = torch.Generator(device=dev).manual_seed(SEED)
for B in (1, 8):
    for L in (2048, 4096):
        KV = [(torch.randn(B, hkv, L, d, device=dev, generator=g, dtype=torch.float16),
               torch.randn(B, hkv, L, d, device=dev, generator=g, dtype=torch.float16)) for _ in range(nl)]
        packed = [pack_layer(K, V) for K, V in KV]
        pack_ms = timeit(lambda: [pack_layer(K, V) for K, V in KV])
        unpack_ms = timeit(lambda: [unpack_layer(pk) for pk in packed])
        ser = sum(nbytes(pk) for pk in packed)
        formula = B * quant_bytes(cfg, L, 4)
        fp16 = B * 2 * nl * hkv * d * L * 2
        tok = torch.randint(1000, 10000, (B, 1), device=dev)

        def make_cache(pairs):
            c = DynamicCache()
            for i, (K, V) in enumerate(pairs):
                c.update(K, V, i)
            return c

        def step(pairs):
            c = make_cache(pairs)
            with torch.no_grad():
                model(input_ids=tok, past_key_values=c, use_cache=True,
                      position_ids=torch.full((B, 1), L, device=dev), cache_position=torch.tensor([L], device=dev))

        torch.cuda.reset_peak_memory_stats()
        fp16_step = timeit(lambda: step(KV))
        peak_fp16 = torch.cuda.max_memory_allocated() / 2**20
        torch.cuda.reset_peak_memory_stats()
        unpack_step = timeit(lambda: step([unpack_layer(pk) for pk in packed]))
        peak_unpack = torch.cuda.max_memory_allocated() / 2**20
        rows.append({"B": B, "L": L, "pack_ms": pack_ms[0], "unpack_ms": unpack_ms[0],
                     "serialized_bytes": ser, "formula_bytes": formula, "fp16_bytes": fp16, "ratio_vs_fp16": ser / fp16,
                     "decode_step_fp16_ms": fp16_step[0], "decode_step_with_unpack_ms": unpack_step[0],
                     "peak_alloc_mb_fp16_step": peak_fp16, "peak_alloc_mb_unpack_step": peak_unpack})
        print(rows[-1], flush=True)
        del KV, packed
        torch.cuda.empty_cache()
df = pd.DataFrame(rows)
df.to_csv(OUT / "packed_benchmark.csv", index=False)
import json  # noqa: E402
(OUT / "packed_check.json").write_text(json.dumps(check, indent=2))
print("Done.", flush=True)
