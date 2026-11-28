"""Task 2 Part A: dense vs. tiled online-softmax vs. PyTorch SDPA causal attention.

  python scripts/t2a_operators.py
Writes results/task2/attention/{checks.json, accuracy.csv, timing.csv}.
B=2, H=4, d=64, L in {256,...,8192}, FP16 and FP32. Errors are against the dense FP32 reference
(outputs and dQ, dK, dV for a fixed random upstream gradient). 3 warm-up + 10 timed CUDA-synchronised
repetitions per condition; out-of-memory and unsupported backends are recorded, not hidden.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.common.utils import RESULTS_DIR, SEED, save_json, set_seed
from src.task2.attention import dense_causal, sdpa_causal, tiled_causal

OUT = RESULTS_DIR / "task2" / "attention"
OUT.mkdir(parents=True, exist_ok=True)
B, H, D = 2, 4, 64
LENGTHS = [256, 512, 1024, 2048, 4096, 8192]
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}
BACKENDS = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "math": SDPBackend.MATH, "cudnn": SDPBackend.CUDNN_ATTENTION}
dev = "cuda"
set_seed(SEED)


def inputs(L, dtype, seed=SEED):
    g = torch.Generator(device=dev).manual_seed(seed + L)
    q, k, v, go = (torch.randn(B, H, L, D, device=dev, generator=g) for _ in range(4))
    return [x.to(dtype) for x in (q, k, v)], go.to(dtype)


def sync():
    torch.cuda.synchronize()


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def run_fb(fn, q, k, v, go):
    q, k, v = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
    o = fn(q, k, v)
    o.backward(go)
    return o.detach(), q.grad, k.grad, v.grad


def method_fn(name):
    if name == "dense":
        return dense_causal
    if name.startswith("tiled"):
        t = int(name.split("_")[1])
        return lambda q, k, v: tiled_causal(q, k, v, tile=t)
    if name == "sdpa_default":
        return sdpa_causal
    be = BACKENDS[name.split("_", 1)[1]]

    def f(q, k, v):
        with sdpa_kernel(be):
            return sdpa_causal(q, k, v)
    return f


def detect_backend(dtype, L=512):
    """Which kernel the default SDPA call actually launches (from CUDA kernel names)."""
    (q, k, v), _ = inputs(L, dtype)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        sdpa_causal(q, k, v)
        sync()
    names = sorted({e.key for e in prof.key_averages()})
    low = " ".join(names).lower()
    if "flash" in low:
        b = "flash"
    elif "fmha" in low or "efficient" in low or "mem_eff" in low:
        b = "efficient"
    elif "cudnn" in low:
        b = "cudnn"
    else:
        b = "math (bmm + softmax)"
    return b, names


# 1. correctness checks -----------------------------------------------------------------
checks = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__}
for dn, dt in DTYPES.items():
    b, names = detect_backend(dt)
    checks[f"sdpa_default_backend_{dn}"] = b
    checks[f"sdpa_default_kernels_{dn}"] = names[:12]
print("default SDPA backend:", {k: v for k, v in checks.items() if "backend" in k}, flush=True)

(q, k, v), _ = inputs(512, torch.float32)
i0 = 300
q2, k2, v2 = q.clone(), k.clone(), v.clone()
k2[:, :, i0 + 1:] += 5 * torch.randn_like(k2[:, :, i0 + 1:])
v2[:, :, i0 + 1:] += 5 * torch.randn_like(v2[:, :, i0 + 1:])
causal = {}
for name in ["dense", "tiled_128", "sdpa_default"]:
    f = method_fn(name)
    a, b2 = f(q, k, v), f(q2, k2, v2)
    causal[name] = {"max_change_at_or_before_i0": float((a[:, :, :i0 + 1] - b2[:, :, :i0 + 1]).abs().max()),
                    "max_change_after_i0": float((a[:, :, i0 + 1:] - b2[:, :, i0 + 1:]).abs().max())}
checks["causality_perturb_future_kv"] = causal
print("causality:", causal, flush=True)

ref = dense_causal(q.double(), k.double(), v.double()).float()
tiles = {}
for t in [16, 64, 128, 256, 512]:
    for dn, dt in DTYPES.items():
        o = tiled_causal(q.to(dt), k.to(dt), v.to(dt), tile=t).float()
        tiles[f"tile{t}_{dn}"] = float((o - ref).abs().max())
checks["tiled_tile_sizes_maxabs_vs_fp64_L512"] = tiles

ext = {}
for scale in [10.0, 30.0]:
    qs, ks = q * scale, k * scale
    r64 = dense_causal(qs.double(), ks.double(), v.double())
    for dn, dt in DTYPES.items():
        for name in ["dense", "tiled_128", "sdpa_default"]:
            try:
                o = method_fn(name)(qs.to(dt), ks.to(dt), v.to(dt)).double()
                ext[f"scale{int(scale)}_{name}_{dn}"] = {"finite": bool(torch.isfinite(o).all()),
                                                         "maxabs_vs_fp64": float((o - r64).abs().max())}
            except Exception as e:  # noqa: BLE001
                ext[f"scale{int(scale)}_{name}_{dn}"] = {"error": str(e)[:80]}
checks["extreme_scores"] = ext
allowed = torch.ones(512, 512, dtype=torch.bool, device=dev).tril()
allowed[5] = False                       # a fully masked query row (padding-style)
od = dense_causal(q, k, v, allowed)
osd = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
checks["fully_masked_row"] = {"dense_row_output": "zeros" if bool((od[:, :, 5] == 0).all()) else "nonzero",
                              "dense_finite": bool(torch.isfinite(od).all()),
                              "sdpa_row_has_nan": bool(torch.isnan(osd[:, :, 5]).any()),
                              "sdpa_row_maxabs": float(torch.nan_to_num(osd[:, :, 5]).abs().max())}
print("edge cases:", checks["fully_masked_row"], flush=True)
save_json(checks, OUT / "checks.json")

# 2. accuracy and timing ------------------------------------------------------------------
METHODS = ["dense", "tiled_128", "sdpa_default", "sdpa_efficient", "sdpa_math", "sdpa_flash", "sdpa_cudnn"]
acc_rows, time_rows = [], []
for L in LENGTHS:
    (q32, k32, v32), go32 = inputs(L, torch.float32)
    try:
        torch.cuda.empty_cache()
        ref = run_fb(dense_causal, q32, k32, v32, go32)
    except Exception as e:  # noqa: BLE001
        if not is_oom(e):
            raise
        ref = None
        print(f"L={L}: FP32 dense reference out of memory; accuracy not computed", flush=True)
    torch.cuda.empty_cache()
    for dn, dt in DTYPES.items():
        (q, k, v), go = inputs(L, dt)
        for name in METHODS:
            f = method_fn(name)
            row = {"L": L, "dtype": dn, "method": name}
            try:
                torch.cuda.empty_cache()
                res = run_fb(f, q, k, v, go)
                if ref is not None:
                    for nm, a, r in zip(["out", "dq", "dk", "dv"], res, ref):
                        d = a.float() - r
                        row[f"{nm}_maxabs"] = float(d.abs().max())
                        row[f"{nm}_rel_fro"] = float(d.norm() / r.norm())
                acc_rows.append(row)
                del res
                # timing: forward only, then forward + backward
                for mode in ["fwd", "fwd_bwd"]:
                    ts = []
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    base_alloc = torch.cuda.memory_allocated()
                    for r_ in range(13):
                        if mode == "fwd":
                            sync(); t0 = time.perf_counter()
                            with torch.no_grad():
                                f(q, k, v)
                            sync()
                        else:
                            qq, kk, vv = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
                            sync(); t0 = time.perf_counter()
                            f(qq, kk, vv).backward(go)
                            sync()
                        if r_ >= 3:
                            ts.append((time.perf_counter() - t0) * 1000)
                    time_rows.append({"L": L, "dtype": dn, "method": name, "mode": mode, "status": "ok",
                                      "ms_mean": sum(ts) / len(ts), "ms_min": min(ts), "ms_max": max(ts), "reps": len(ts),
                                      "peak_alloc_mb": (torch.cuda.max_memory_allocated() - base_alloc) / 2**20,
                                      "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20})
                print(f"L={L} {dn} {name}: fwd {time_rows[-2]['ms_mean']:.2f} ms, fwd+bwd {time_rows[-1]['ms_mean']:.2f} ms,"
                      f" peak {time_rows[-1]['peak_alloc_mb']:.0f} MB" + (f", out err {row.get('out_maxabs', float('nan')):.2e}"), flush=True)
            except Exception as e:  # noqa: BLE001
                status = "out_of_memory" if is_oom(e) else "unsupported: " + str(e).splitlines()[0][:70]
                time_rows.append({"L": L, "dtype": dn, "method": name, "mode": "fwd_bwd", "status": status})
                print(f"L={L} {dn} {name}: {status}", flush=True)
                torch.cuda.empty_cache()
        pd.DataFrame(acc_rows).to_csv(OUT / "accuracy.csv", index=False)
        pd.DataFrame(time_rows).to_csv(OUT / "timing.csv", index=False)
    del ref
print("Done. Saved results/task2/attention/", flush=True)
