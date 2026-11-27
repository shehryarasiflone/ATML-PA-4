"""Group-wise asymmetric weight-only quantization (Q1/Q2/Q3 share these conventions exactly).

Conventions (fixed for every quantized condition):
  * groups of 128 consecutive input columns within each output row (1536 and 8960 are multiples
    of 128, so no padding is needed; the assert enforces it);
  * per group: scale = (max - min) / (2^b - 1), stored as FP16; zero-point z = round(-min / scale),
    clamped to [0, 2^b - 1] and stored as a b-bit unsigned integer;
  * q = clamp(round(w / scale) + z, 0, 2^b - 1) with round-half-to-even (torch.round); no clipping search;
  * dequantized weight = (q - z) * scale, using the FP16-rounded scale (what a loader would see);
  * packing: b-bit codes stored contiguously, little-endian within bytes, per matrix, rounded up
    to whole bytes; zero-points packed the same way; scales as FP16.
The model is evaluated with dequantized FP16 weights (fake quantization, dense FP16 kernels):
packed bytes are an exact storage calculation, not a measured low-bit execution.
"""
import math

import torch

GROUP = 128


@torch.no_grad()
def fake_quant(W, bits, group=GROUP):
    out, din = W.shape
    assert din % group == 0, "input dimension must be a multiple of the group size"
    Wg = W.float().reshape(out, din // group, group)
    wmin, wmax = Wg.amin(-1, keepdim=True), Wg.amax(-1, keepdim=True)
    qmax = 2 ** bits - 1
    scale = ((wmax - wmin) / qmax).clamp(min=1e-5).half().float()
    zero = torch.round(-wmin / scale).clamp(0, qmax)
    q = torch.clamp(torch.round(Wg / scale) + zero, 0, qmax)
    return ((q - zero) * scale).reshape(out, din)


def packed_bytes(out, din, bits, group=GROUP):
    """Bytes for one matrix: codes + FP16 scales + b-bit zero-points."""
    n_groups = out * (din // group)
    return math.ceil(out * din * bits / 8) + 2 * n_groups + math.ceil(n_groups * bits / 8)


def model_packed_bytes(shapes_bits, other_params, extra_fp16=0):
    """shapes_bits: iterable of (out, din, bits) for the 196 eligible matrices;
    other_params: non-eligible parameters kept in FP16 (tied once); extra_fp16: serialized FP16 values
    (e.g. AWQ o_proj compensation scales)."""
    elig = sum(packed_bytes(o, i, b) for o, i, b in shapes_bits)
    return {"eligible_packed_bytes": elig, "other_fp16_bytes": 2 * other_params,
            "awq_serialized_scale_bytes": 2 * extra_fp16, "total_bytes": elig + 2 * other_params + 2 * extra_fp16}
