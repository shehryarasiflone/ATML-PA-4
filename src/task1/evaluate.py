"""Shared Task 1 evaluation: two perplexity suites, HellaSwag, and resource measurements.

Every condition (baseline, pruning, quantization) is evaluated by `evaluate_condition`,
so all models see exactly the same windows, scored tokens and HellaSwag candidates.
"""
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.common.data import data_root, records
from src.common.utils import RESULTS_DIR, save_json

T1_DIR = RESULTS_DIR / "task1"
PPL_SUITES = {"wikitext2": "wikitext2_test_windows_locked.jsonl.gz",
              "c4val": "c4_validation_windows_locked.jsonl.gz"}
HS_FILE = "hellaswag_val_locked.jsonl.gz"
HS_DIAG_N = 512
HS_CONVENTION = ("lm-evaluation-harness hellaswag (v1.0) prompt: preprocess(activity_label + ': ' + ctx_a + ' ' "
                 "+ ctx_b.capitalize()); continuation ' ' + preprocess(ending); zero-shot; acc = argmax summed "
                 "log-likelihood; acc_norm = argmax log-likelihood / len(ending in characters)")


# --------------------------------------------------------------------------- data
def load_windows(suite):
    return records(data_root() / "public_locked" / PPL_SUITES[suite])


def load_dev_windows():
    """The 32 C4 development sequences as perplexity windows (511 scored tokens each).
    Used only for selecting allocations (P3/Q3) and calibration-vs-held-out diagnostics."""
    rows = records(data_root() / "public_locked" / "c4_development_locked.jsonl.gz")
    return [{"token_ids": r["token_ids"], "score_target_offsets": [1, len(r["token_ids"])],
             "scored_tokens": len(r["token_ids"]) - 1, "source_row": r["source_row"], "token_offset": 0}
            for r in rows]


def load_calibration(which="calibration"):
    """128 (calibration) or 32 (development) C4 sequences of 512 tokens -> LongTensor [n, 512]."""
    rows = records(data_root() / "public_locked" / f"c4_{which}_locked.jsonl.gz")
    return torch.tensor([r["token_ids"] for r in rows], dtype=torch.long), [r["source_row"] for r in rows]


def _hs_pre(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def load_hellaswag(tok, n=None):
    """List of dicts with context/continuation token ids, following lm-eval's _encode_pair."""
    rows = records(data_root() / "public_locked" / HS_FILE)
    rows = sorted(rows, key=lambda r: r["order"])[: n or None]
    out = []
    for r in rows:
        ex = r["example"]
        ctx = _hs_pre(ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize())
        endings = [_hs_pre(e) for e in ex["endings"]]
        ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"]
        conts = []
        for e in endings:
            whole = tok(ctx + " " + e, add_special_tokens=False)["input_ids"]
            conts.append(whole[len(ctx_ids):])
        out.append({"order": r["order"], "row": r["source_row_index"], "gold": int(ex["label"]),
                    "ctx_ids": ctx_ids, "cont_ids": conts, "char_len": [len(e) for e in endings]})
    return out


# --------------------------------------------------------------------------- perplexity
@torch.no_grad()
def perplexity(model, windows, batch=8):
    """Pooled token NLL over fixed windows. Returns (summary, per-window DataFrame)."""
    dev = next(model.parameters()).device
    rows = []
    for i in range(0, len(windows), batch):
        chunk = windows[i:i + batch]
        L = max(len(w["token_ids"]) for w in chunk)
        ids = torch.zeros(len(chunk), L, dtype=torch.long)
        att = torch.zeros(len(chunk), L, dtype=torch.long)
        for j, w in enumerate(chunk):
            ids[j, :len(w["token_ids"])] = torch.tensor(w["token_ids"])
            att[j, :len(w["token_ids"])] = 1
        logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev), use_cache=False).logits
        for j, w in enumerate(chunk):
            lo, hi = w["score_target_offsets"]           # targets are token_ids[lo:hi]
            tgt = torch.tensor(w["token_ids"][lo:hi], device=dev)
            lp = F.cross_entropy(logits[j, lo - 1:hi - 1].float(), tgt, reduction="sum")
            assert hi - lo == w["scored_tokens"]
            rows.append({"window": i + j, "source_row": w["source_row"], "token_offset": w["token_offset"],
                         "nll_sum": lp.item(), "n": hi - lo})
        del logits
    df = pd.DataFrame(rows)
    N = int(df.n.sum())
    nll = df.nll_sum.sum() / N
    return {"scored_tokens": N, "mean_nll": float(nll), "ppl": float(np.exp(nll)), "windows": len(df)}, df


# --------------------------------------------------------------------------- hellaswag
@torch.no_grad()
def hellaswag(model, examples, batch_examples=8):
    """Zero-shot 4-way scoring. Returns (summary, per-example DataFrame with all candidate scores)."""
    dev = next(model.parameters()).device
    rows = []
    for i in range(0, len(examples), batch_examples):
        chunk = examples[i:i + batch_examples]
        seqs, spans = [], []
        for ex in chunk:
            for c in ex["cont_ids"]:
                s = ex["ctx_ids"] + c
                seqs.append(s)
                spans.append((len(ex["ctx_ids"]), len(s)))
        L = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), L, dtype=torch.long)
        att = torch.zeros(len(seqs), L, dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s)
            att[j, :len(s)] = 1
        logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev), use_cache=False).logits
        lls = []
        for j, (a, b) in enumerate(spans):
            tgt = ids[j, a:b].to(dev)
            lls.append(-F.cross_entropy(logits[j, a - 1:b - 1].float(), tgt, reduction="sum").item())
        for k, ex in enumerate(chunk):
            ll = np.array(lls[4 * k:4 * k + 4])
            norm = ll / np.array(ex["char_len"], dtype=float)
            rows.append({"order": ex["order"], "row": ex["row"], "gold": ex["gold"],
                         **{f"ll{c}": ll[c] for c in range(4)}, **{f"chars{c}": ex["char_len"][c] for c in range(4)},
                         "pred": int(ll.argmax()), "pred_norm": int(norm.argmax())})
        del logits
    df = pd.DataFrame(rows)
    df["correct"] = (df.pred == df.gold).astype(int)
    df["correct_norm"] = (df.pred_norm == df.gold).astype(int)
    return {"n": len(df), "acc": float(df.correct.mean()), "acc_norm": float(df.correct_norm.mean())}, df


# --------------------------------------------------------------------------- resources
def _sync():
    torch.cuda.synchronize()


@torch.no_grad()
def latency(model, reps=5, warmup=2, prompt_lens=(512, 2048), decode_tokens=64, seed=6304):
    """Dense execution timing (ms). Prefill = one forward over the prompt; decode = 64 greedy
    single-token steps with the KV cache after a 512-token prefill (prefill excluded)."""
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    out = {"reps": reps, "warmup": warmup, "unit": "ms"}
    for L in prompt_lens:
        x = torch.randint(1000, 100000, (1, L), generator=g).to(dev)
        ts = []
        for r in range(warmup + reps):
            _sync(); t0 = time.perf_counter()
            model(input_ids=x, use_cache=True)
            _sync(); t = (time.perf_counter() - t0) * 1000
            if r >= warmup:
                ts.append(t)
        out[f"prefill_{L}_ms_mean"] = float(np.mean(ts))
        out[f"prefill_{L}_ms_std"] = float(np.std(ts))
    x = torch.randint(1000, 100000, (1, 512), generator=g).to(dev)
    ts = []
    for r in range(warmup + reps):
        o = model(input_ids=x, use_cache=True)
        pkv, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
        _sync(); t0 = time.perf_counter()
        for _ in range(decode_tokens):
            o = model(input_ids=nxt, past_key_values=pkv, use_cache=True)
            pkv, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
        _sync(); t = (time.perf_counter() - t0) * 1000
        if r >= warmup:
            ts.append(t)
    out[f"decode_{decode_tokens}_ms_mean"] = float(np.mean(ts))
    out[f"decode_{decode_tokens}_ms_std"] = float(np.std(ts))
    out["decode_ms_per_token"] = out[f"decode_{decode_tokens}_ms_mean"] / decode_tokens
    return out


# --------------------------------------------------------------------------- one condition
def dev_perplexity(model):
    s, _ = perplexity(model, load_dev_windows())
    return s


def evaluate_condition(model, tok, tag, hs="full", timing=True, extra=None, hs_cache=None, dev=True):
    """Evaluate one model and save everything under results/task1/conditions/<tag>/.

    hs: 'full' (2,000 examples), 'diag' (first 512) or None.
    """
    d = T1_DIR / "conditions" / tag
    d.mkdir(parents=True, exist_ok=True)
    summary = {"tag": tag, **(extra or {})}
    torch.cuda.reset_peak_memory_stats()
    for suite in PPL_SUITES:
        t0 = time.time()
        s, df = perplexity(model, load_windows(suite))
        df.to_csv(d / f"ppl_{suite}.csv", index=False)
        summary[suite] = {**s, "eval_sec": round(time.time() - t0, 1)}
        print(f"[{tag}] {suite}: ppl {s['ppl']:.3f}  nll {s['mean_nll']:.4f}  N={s['scored_tokens']}", flush=True)
    if dev:
        summary["c4dev"] = dev_perplexity(model)
        print(f"[{tag}] c4dev (selection set): ppl {summary['c4dev']['ppl']:.3f}", flush=True)
    if hs:
        t0 = time.time()
        exs = hs_cache if hs_cache is not None else load_hellaswag(tok, None if hs == "full" else HS_DIAG_N)
        if hs == "diag":
            exs = exs[:HS_DIAG_N]
        s, df = hellaswag(model, exs)
        df.to_csv(d / f"hellaswag_{hs}.csv", index=False)
        summary[f"hellaswag_{hs}"] = {**s, "eval_sec": round(time.time() - t0, 1), "convention": HS_CONVENTION}
        print(f"[{tag}] hellaswag_{hs}: acc {s['acc']:.4f}  acc_norm {s['acc_norm']:.4f}  n={s['n']}", flush=True)
    summary["eval_peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    if timing:
        torch.cuda.reset_peak_memory_stats()
        summary["latency"] = latency(model)
        summary["latency"]["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        lt = summary["latency"]
        print(f"[{tag}] prefill512 {lt['prefill_512_ms_mean']:.1f} ms | prefill2048 {lt['prefill_2048_ms_mean']:.1f} ms"
              f" | decode64 {lt['decode_64_ms_mean']:.1f} ms | peak {lt['peak_alloc_gb']} GB", flush=True)
    save_json(summary, d / "summary.json")
    return summary
