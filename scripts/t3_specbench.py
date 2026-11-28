"""Task 3 steps 2 and 6: EOS-aware speculative benchmark of one draft against target-only decoding.

  python scripts/t3_specbench.py --draft original --rep 0
  python scripts/t3_specbench.py --draft original --rep 1 --costs --fixed64
--draft: 'original' or a merged checkpoint folder (results of the distillation step).
--adapter results/task3/distill/<run>/adapter: the same draft unmerged (PEFT wrapper), to measure
wrapper overhead only; it is not the primary speed comparison.
For each of the 120 held-out prompts and each repetition, target-only, gamma=2 and gamma=4 run
adjacently in a seeded counterbalanced order with the same sampling seed (temperature 1, top-p 1,
64 new tokens, EOS stops). Per run: accepted / examined / proposed, accepted prefix per verification,
target passes per output token, draft passes, EOS/length flags, timers split into prefill, draft
proposals, target verification, accept/residual sampling, cache rollback and host overhead.
--costs: isolated single-step draft/target decode and (gamma+1)-token verification cost at several
context lengths -> c and v_gamma.  --fixed64: 24 math prompts, EOS ignored, 64 tokens (throughput).
Writes results/task3/spec/<draft>/rep<k>.csv (+ costs.csv, fixed64.csv).
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.common.utils import RESULTS_DIR, SEED, set_seed
from src.task3.data import DRAFT, DRAFT_REV, TARGET, TARGET_REV, chat_ids, eos_ids, eval_prompts, load, tokenizer
from src.task3.specdec import HFEngine, speculative_generate, target_only_generate

p = argparse.ArgumentParser()
p.add_argument("--draft", default="original")
p.add_argument("--rep", type=int, default=0)
p.add_argument("--costs", action="store_true")
p.add_argument("--fixed64", action="store_true")
p.add_argument("--max-new", type=int, default=64)
p.add_argument("--adapter", help="unmerged LoRA adapter folder (wrapper-overhead timing only)")
args = p.parse_args()
set_seed(SEED)
dev = "cuda"
tok = tokenizer()
EOS = eos_ids(tok)
tm = load(TARGET, TARGET_REV)
if args.adapter:
    from peft import PeftModel                     # FP16 backbone + FP32 LoRA path, as in training
    dm = PeftModel.from_pretrained(load(DRAFT, DRAFT_REV), args.adapter).eval()
    dname = "unmerged_" + Path(args.adapter).parent.name
elif args.draft == "original":
    dm = load(DRAFT, DRAFT_REV)
    dname = "original"
else:
    from transformers import AutoModelForCausalLM
    dm = AutoModelForCausalLM.from_pretrained(args.draft, dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()
    dname = Path(args.draft).name
OUT = RESULTS_DIR / "task3" / "spec" / dname
OUT.mkdir(parents=True, exist_ok=True)
prompts = eval_prompts()


def run(method, ids, seed, max_new, eos):
    gen = torch.Generator(device=dev).manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    if method == "target":
        out, st = target_only_generate(HFEngine(tm), ids, max_new, gen, eos=eos, timers=True)
    else:
        out, st = speculative_generate(HFEngine(tm), HFEngine(dm), ids, int(method[1:]), max_new, gen, eos=eos, timers=True)
    torch.cuda.synchronize(); wall = time.perf_counter() - t0
    row = {"method": method, "n_out": len(out), "eos": st["eos"], "length_capped": (not st["eos"]) and len(out) >= max_new,
           "wall_s": wall, "peak_alloc_mb": torch.cuda.max_memory_allocated() / 2**20, "prompt_len": ids.shape[1],
           "target_passes": st["target_passes"], "t_prefill": st["t_prefill"]}
    if method == "target":
        row["t_decode"] = st["t_decode"]
    else:
        ap = st["accepted_prefix"]
        row.update({k: st[k] for k in ["rounds", "proposed", "examined", "accepted", "bonus", "draft_passes",
                                       "t_draft", "t_target", "t_accept", "t_rollback"]})
        row["mean_accepted_prefix"] = sum(ap) / len(ap)
        row["t_host_other"] = wall - st["t_prefill"] - st["t_draft"] - st["t_target"] - st["t_accept"] - st["t_rollback"]
    row["tokens_per_s"] = len(out) / wall
    row["target_passes_per_token"] = st["target_passes"] / max(1, len(out))
    return row, out


# warm-up (excluded): CUDA kernels, allocator, both models
for m in ["target", "g2", "g4"]:
    run(m, torch.tensor([chat_ids(tok, prompts[0]["text"])], device=dev), 1, 16, EOS)

rows = []
rng = random.Random(f"order:{SEED}:{args.rep}:{dname}")
for i, pr in enumerate(prompts):
    ids = torch.tensor([chat_ids(tok, pr["text"])], device=dev)
    seed = SEED * 1000 + 100 * args.rep + i
    order = ["target", "g2", "g4"]
    rng.shuffle(order)
    for pos, m in enumerate(order):
        row, out = run(m, ids, seed, args.max_new, EOS)
        row.update({"prompt_id": pr["id"], "domain": pr["domain"], "rep": args.rep, "seed": seed, "order_pos": pos,
                    "draft": dname, "text": tok.decode(out, skip_special_tokens=True)})
        rows.append(row)
    if (i + 1) % 20 == 0:
        df = pd.DataFrame(rows)
        sp = df[df.method != "target"].groupby("method")
        print(f"[{dname} rep{args.rep}] {i + 1}/120 | acc. rate " +
              " ".join(f"{m}:{g.accepted.sum() / g.examined.sum():.3f}" for m, g in sp) +
              " | tok/s " + " ".join(f"{m}:{g.tokens_per_s.mean():.1f}" for m, g in df.groupby("method")), flush=True)
        df.to_csv(OUT / f"rep{args.rep}.csv", index=False)
pd.DataFrame(rows).to_csv(OUT / f"rep{args.rep}.csv", index=False)

if args.fixed64:
    fr = []
    for i, pr in enumerate([q for q in prompts if q["domain"] == "math"][:24]):
        ids = torch.tensor([chat_ids(tok, pr["text"])], device=dev)
        for m in ["target", "g2", "g4"]:
            row, _ = run(m, ids, SEED * 1000 + 900 + i, 64, ())
            row.update({"prompt_id": pr["id"], "draft": dname})
            fr.append(row)
    pd.DataFrame(fr).to_csv(OUT / "fixed64.csv", index=False)
    print("fixed64:", pd.DataFrame(fr).groupby("method").tokens_per_s.mean().to_dict(), flush=True)

if args.costs:
    cr = []
    base = chat_ids(tok, prompts[0]["text"])
    for L in [64, 256, 512, 1024]:
        ids = (base * (L // len(base) + 1))[:L]
        for who, model in [("target", tm), ("draft", dm)]:
            for n_tok in [1, 3, 5]:                    # single step, gamma=2 and gamma=4 verification (pending + gamma)
                e = HFEngine(model)
                e.prefill(torch.tensor([ids], device=dev))
                ts = []
                for r in range(13):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    e.extend(list(range(1000, 1000 + n_tok)))
                    torch.cuda.synchronize()
                    if r >= 3:
                        ts.append(1000 * (time.perf_counter() - t0))
                    e.rollback(L)
                cr.append({"context": L, "model": who, "tokens": n_tok, "ms_mean": sum(ts) / len(ts), "ms_min": min(ts)})
    c = pd.DataFrame(cr)
    c.to_csv(OUT / "costs.csv", index=False)
    print(c.pivot_table(index="context", columns=["model", "tokens"], values="ms_mean").round(2), flush=True)
print("Done.", flush=True)
