"""Task 2 Part C (C1-C5 quality part): KV-cache policies on the 180 retrieval prompts.

  python scripts/t2c_kv.py                 # all prompts (resumes from results/task2/kv/rows.csv)
  python scripts/t2c_kv.py --limit 6       # smoke test
Policies (context cache only, all at the case-specific INT4 channel-wise-K/token-wise-V byte budget):
  full            unchanged FP16 reference
  sink_recent     FP16 eviction: 8 sink + most recent positions (64-recent safeguard included)
  question        FP16 eviction: 8 sink + 64 recent + highest question-attention positions
  int4_tokenwise  INT4, token-wise K and V (groups of 64 features)
  int4_kchan      INT4, channel-wise K (64 tokens per channel) / token-wise V   <- budget reference
  hybrid          8 sink + 64 recent + top question positions in FP16, rest INT2 channel-K/token-V
No retention parameter was tuned (sink/recent sizes are fixed by the guide), so no development
prompts are used; the importance score never sees the code or the evidence span.
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.data import data_root, records
from src.common.utils import RESULTS_DIR, SEED, save_json, set_seed
from src.task2.kvcache import (BlockQuantizer, clone_cache, eviction_budget_positions, forward_with_cache,
                               fp16_pos_bytes, greedy_decode, hybrid_bytes, quant_bytes, quantize_context,
                               quantize_positions, question_scores, select_positions)

TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_REV = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SINK, RECENT, MAX_NEW = 8, 64, 8
POLICIES = ["full", "sink_recent", "question", "int4_tokenwise", "int4_kchan", "hybrid"]
OUT = RESULTS_DIR / "task2" / "kv"
OUT.mkdir(parents=True, exist_ok=True)

p = argparse.ArgumentParser()
p.add_argument("--limit", type=int, default=None)
args = p.parse_args()
set_seed(SEED)
dev = "cuda"
tok = AutoTokenizer.from_pretrained(TARGET, revision=TARGET_REV)
model = AutoModelForCausalLM.from_pretrained(TARGET, revision=TARGET_REV, dtype=torch.float16,
                                             attn_implementation="sdpa").to(dev).eval()
cfg = model.config
EOS = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id, tok.convert_tokens_to_ids("<|endoftext|>")}
cases = records(data_root() / "kv_retrieval" / "kv_retrieval_cases.jsonl.gz")
cases = cases[: args.limit] if args.limit else cases


def encode(case):
    """Chat-template the prompt; split at QUESTION_BEGIN on a token boundary; map evidence chars to tokens."""
    text = tok.apply_chat_template([{"role": "user", "content": case["user_text"]}], tokenize=False,
                                   add_generation_prompt=True)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    qchar = text.index("QUESTION_BEGIN")
    L = next(i for i, (s, e) in enumerate(offs) if e > qchar)         # first token touching the question
    u0 = text.index(case["user_text"])
    es, ee = u0 + case["evidence_char_start"], u0 + case["evidence_char_end"]
    ev = [i for i, (s, e) in enumerate(offs) if s < ee and e > es]
    assert max(ev) < L, "evidence must lie in the context"
    return torch.tensor([ids], device=dev), L, ev


def extract(text):
    m = re.search(r"\d{5}", text)
    return m.group(0) if m else None


def sync():
    torch.cuda.synchronize()


@torch.no_grad()
def answer_metrics(cache, suffix, L, gold_ids, ref_first=None, ref_tf=None, quantizer=None):
    """Append the question suffix, then (a) teacher-forced gold answer on a cache copy, (b) greedy decode."""
    sync(); t0 = time.perf_counter()
    lg = forward_with_cache(model, suffix, cache, L, quantizer)
    sync(); t_suffix = time.perf_counter() - t0
    first = lg[0, -1].float()
    c2 = clone_cache(cache)
    q2 = None if quantizer is None else BlockQuantizer(quantizer.layout, quantizer.bits, quantizer.start)
    tf = forward_with_cache(model, gold_ids[:, :-1], c2, L + suffix.shape[1], q2)[0].float()
    del c2
    tf_logits = torch.cat([first[None], tf], 0)                          # predicts every gold token
    nll = float(F.cross_entropy(tf_logits, gold_ids[0], reduction="mean"))
    sync(); t0 = time.perf_counter()
    gen = greedy_decode(model, cache, lg, L + suffix.shape[1], MAX_NEW, EOS, quantizer)
    sync(); t_dec = time.perf_counter() - t0
    out = {"nll_gold": nll, "suffix_ms": 1000 * t_suffix, "decode_ms": 1000 * t_dec, "gen_tokens": len(gen),
           "gen_text": tok.decode(gen, skip_special_tokens=True), "finite": bool(torch.isfinite(tf_logits).all())}
    if ref_first is not None:
        lp, lq = F.log_softmax(ref_first, -1), F.log_softmax(first, -1)
        out["kl_first"] = float((lp.exp() * (lp - lq)).sum())
        rp, rq = F.log_softmax(ref_tf, -1), F.log_softmax(tf_logits, -1)
        out["kl_tf_mean"] = float((rp.exp() * (rp - rq)).sum(-1).mean())
    return out, first, tf_logits, gen


# ---------------------------------------------------------------- C1 validation (first prompts)
val = {}
ids, L, _ = encode(cases[0])
with torch.no_grad():
    full_logits = model(input_ids=ids).logits[0, -1].float()
    out = model(input_ids=ids[:, :L], use_cache=True)
    cache = out.past_key_values
    split_logits = forward_with_cache(model, ids[:, L:], cache, L)[0, -1].float()
val["full_vs_split_prefill_max_abs_logit_diff"] = float((full_logits - split_logits).abs().max())
val["full_vs_split_argmax_equal"] = bool(full_logits.argmax() == split_logits.argmax())
with torch.no_grad():
    ref_gen = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)[0, ids.shape[1]:].tolist()
    c = model(input_ids=ids[:, :L], use_cache=True).past_key_values
    select_positions(c, torch.arange(L, device=dev))                     # 100% retention through the edit path
    lg = forward_with_cache(model, ids[:, L:], c, L)
    our_gen = greedy_decode(model, c, lg, ids.shape[1], MAX_NEW, EOS)
val["retain_100pct_generation_equal_to_hf_generate"] = ref_gen[:len(our_gen)] == our_gen
val["hf_generate_text"], val["our_text"] = tok.decode(ref_gen), tok.decode(our_gen)
print("C1 validation:", val, flush=True)
save_json(val, OUT / "c1_validation.json")

# ---------------------------------------------------------------- main loop
rf = OUT / "rows.csv"
rows = pd.read_csv(rf).to_dict("records") if rf.exists() else []
done = {r["id"] for r in rows if r["policy"] == POLICIES[-1]}
masks = {}
for ci, case in enumerate(cases):
    if case["id"] in done:
        continue
    ids, L, ev = encode(case)
    suffix = ids[:, L:]
    gold = tok(case["code"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
    budget = quant_bytes(cfg, L, 4)
    full_bytes = L * fp16_pos_bytes(cfg)
    torch.cuda.reset_peak_memory_stats()
    sync(); t0 = time.perf_counter()
    with torch.no_grad():
        base = model(input_ids=ids[:, :L], use_cache=True).past_key_values
    sync(); prefill_ms = 1000 * (time.perf_counter() - t0)
    prefill_peak = torch.cuda.max_memory_allocated() / 2**20
    sync(); t0 = time.perf_counter()
    score = question_scores(model, base, suffix, L)
    sync(); score_ms = 1000 * (time.perf_counter() - t0)
    order = torch.argsort(score, descending=True)
    ev_set = set(ev)
    ref_first = ref_tf = None
    for pol in POLICIES:
        torch.cuda.reset_peak_memory_stats()
        sync(); t0 = time.perf_counter()
        cache = clone_cache(base)
        quantizer, kept, relK, relV, byt, n_fp16 = None, None, None, None, full_bytes, L
        if pol == "sink_recent":
            n = min(L, eviction_budget_positions(cfg, budget, 0, fixed_index_bytes=4))
            kept = torch.cat([torch.arange(SINK), torch.arange(L - (n - SINK), L)]).to(dev)
            byt, n_fp16 = n * fp16_pos_bytes(cfg) + 4, n
        elif pol == "question":
            n = min(L, eviction_budget_positions(cfg, budget, 2))
            must = torch.cat([torch.arange(SINK), torch.arange(L - RECENT, L)]).to(dev)
            rest = order[~torch.isin(order, must)][: n - len(must)]
            kept = torch.sort(torch.cat([must, rest])).values
            byt, n_fp16 = n * (fp16_pos_bytes(cfg) + 2), n
        elif pol.startswith("int4"):
            relK, relV = quantize_context(cache, L, "tokenwise" if pol == "int4_tokenwise" else "kchannel_vtoken", 4)
            quantizer = BlockQuantizer("tokenwise" if pol == "int4_tokenwise" else "kchannel_vtoken", 4, (L // 64) * 64)
            byt, n_fp16 = quant_bytes(cfg, L, 4), L % 64
        elif pol == "hybrid":
            must = torch.cat([torch.arange(SINK), torch.arange(L - RECENT, L)]).to(dev)
            ranked = order[~torch.isin(order, must)]
            s = len(ranked)
            while s >= 0 and hybrid_bytes(cfg, L, len(must) + s)[0] > budget:
                s -= 1
            fp_sel = torch.cat([must, ranked[:s]])
            low = torch.sort(ranked[s:]).values
            n_q = (len(low) // 64) * 64
            relK, relV = quantize_positions(cache, low[:n_q], 2) if n_q else (0.0, 0.0)
            kept = torch.sort(torch.cat([fp_sel, low[n_q:]])).values          # FP16 positions
            byt, _, n_fp16 = hybrid_bytes(cfg, L, len(must) + s)
        if kept is not None and pol != "hybrid":
            select_positions(cache, kept)
        sync(); compress_ms = 1000 * (time.perf_counter() - t0)
        m, first, tf_logits, gen = answer_metrics(cache, suffix, L, gold, ref_first, ref_tf, quantizer)
        if pol == "full":
            ref_first, ref_tf = first, tf_logits
        pred = extract(m["gen_text"])
        row = {"id": case["id"], "ctx_target": case["target_tokens"], "position": case["position"],
               "distractors": case["distractors"], "policy": pol, "L_ctx": L, "suffix_len": suffix.shape[1],
               "budget_bytes": budget, "bytes": byt, "frac_fp16_bytes": byt / full_bytes, "within_budget": byt <= budget or pol == "full",
               "n_fp16_positions": n_fp16, "relK": relK, "relV": relV, "pred": pred, "correct": int(pred == case["code"]),
               "evidence_tokens": len(ev), "prefill_ms": prefill_ms, "prefill_peak_mb": prefill_peak,
               "score_ms": score_ms if pol in ("question", "hybrid") else 0.0, "compress_ms": compress_ms,
               "policy_peak_mb": torch.cuda.max_memory_allocated() / 2**20, **m}
        if kept is not None:
            row["evidence_retained_fp16"] = len(ev_set & set(kept.tolist())) / len(ev)
            if ci < 12:
                masks[f"{case['id']}|{pol}"] = kept.cpu().numpy().astype(np.int16)
        rows.append(row)
        del cache
    if ci < 12:
        masks[f"{case['id']}|score"] = score.cpu().numpy().astype(np.float32)
        masks[f"{case['id']}|evidence"] = np.array(ev, dtype=np.int16)
    pd.DataFrame(rows).to_csv(rf, index=False)
    acc = {pp: np.mean([r["correct"] for r in rows if r["policy"] == pp]) for pp in POLICIES}
    print(f"[{ci + 1}/{len(cases)}] {case['id']} L={L} | " + " ".join(f"{k}:{v:.2f}" for k, v in acc.items()), flush=True)
    del base
    torch.cuda.empty_cache()
if masks:
    np.savez_compressed(OUT / f"selection_masks_{len(rows)}.npz", **masks)
print("Done.", flush=True)
