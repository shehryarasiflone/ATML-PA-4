"""Storage accounting for Task 1 (bytes). Tied parameters are counted once.

Dense FP16: 2 bytes per parameter.
Unstructured sparse (guide's model): per eligible matrix, one mask bit per weight (rounded up to whole
  bytes) + 16 bits per nonzero; every non-eligible parameter stays dense FP16.
2:4 semi-structured: per eligible matrix, 16 bits per kept value + 2-bit in-group index per kept value
  (rounded up to whole bytes), i.e. the usual compressed 2:4 layout; shown next to the mask format.
"""
import math

from src.task1.model_utils import eligible_linears, unique_params


def _other_params(model):
    elig_ptrs = {m.weight.data_ptr() for _, m in eligible_linears(model)}
    return sum(p.numel() for _, p in unique_params(model) if p.data_ptr() not in elig_ptrs)


def dense_fp16_bytes(model):
    return 2 * sum(p.numel() for _, p in unique_params(model))


def sparse_mask_bytes(model):
    elig = 0
    for _, m in eligible_linears(model):
        n, nnz = m.weight.numel(), int((m.weight != 0).sum())
        elig += math.ceil(n / 8) + 2 * nnz
    return {"eligible_bytes": elig, "other_fp16_bytes": 2 * _other_params(model),
            "total_bytes": elig + 2 * _other_params(model)}


def nm24_bytes(model):
    elig = 0
    for _, m in eligible_linears(model):
        nnz = int((m.weight != 0).sum())
        elig += 2 * nnz + math.ceil(2 * nnz / 8)
    return {"eligible_bytes": elig, "other_fp16_bytes": 2 * _other_params(model),
            "total_bytes": elig + 2 * _other_params(model)}
