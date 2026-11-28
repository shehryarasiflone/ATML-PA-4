"""Task 3 steps 3-4: off-policy and on-policy forward-KL distillation of the 0.5B draft (LoRA).

  python scripts/t3_distill.py --method offpolicy --seed 6304
  python scripts/t3_distill.py --method onpolicy  --seed 6304
Fixed settings (guide): 40,000 teacher-scored response tokens per run (30,000 math + 10,000 short),
response caps 192 (math) / 32 (short), temperature 1, top-p 1; microbatch 1 response; one optimizer
update per 256 teacher-scored tokens (a response that crosses a boundary is split: its first part
closes the current update, the rest opens the next; the final update may be short); AdamW lr 3e-4,
wd 0.01 (constant); LoRA r=8, alpha=16, dropout 0 on q/k/v/o; FP16 frozen backbones, FP32 adapters
and optimizer state, FP16 autocast with dynamic loss scaling, unscale before clipping at 1.0.
Loss: forward KL(p_teacher || q_student) over the full 151,936-entry vocabulary at every scored
response position, normalised by the scored tokens of the update. Sampled tokens are never used
as hard targets.
Prompt schedule (identical for both methods): seeded permutation within each category, round-robin
reuse after a category is exhausted; the next presentation comes from the category that is furthest
behind its quota share. Off-policy: the whole teacher-response collection for this seed is generated
first (frozen), then trained on in order. On-policy: each response is sampled from the current
student, then scored by the teacher. Every presentation is logged (source id, rollout seed, response
ids, termination reason, scored tokens).
Outputs: results/task3/distill/<method>_s<seed>/{presentations.jsonl, updates.csv, summary.json, adapter/}
Resumable: a checkpoint is written every 20 updates.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, get_peft_model_state_dict

from src.common.utils import RESULTS_DIR, save_json, set_seed
from src.task3.data import DRAFT, DRAFT_REV, TARGET, TARGET_REV, chat_ids, eos_ids, load, tokenizer, train_prompts
from src.task3.specdec import HFEngine, target_only_generate

QUOTA = {"math": 30000, "short": 10000}
CAP = {"math": 192, "short": 32}
UPDATE_TOKENS = 256

p = argparse.ArgumentParser()
p.add_argument("--method", choices=["offpolicy", "onpolicy"], required=True)
p.add_argument("--seed", type=int, required=True)
args = p.parse_args()
dev = "cuda"
OUT = RESULTS_DIR / "task3" / "distill" / f"{args.method}_s{args.seed}"
OUT.mkdir(parents=True, exist_ok=True)
set_seed(args.seed)
tok = tokenizer()
EOS = eos_ids(tok)
math_p, short_p = train_prompts()
pools = {"math": math_p, "short": short_p}
orders = {c: random.Random(f"schedule:{args.seed}:{c}").sample(range(len(v)), len(v)) for c, v in pools.items()}


def schedule():
    """Infinite deterministic presentation schedule: (index, category, prompt, rollout_seed)."""
    done, ptr, i = {"math": 0, "short": 0}, {"math": 0, "short": 0}, 0
    while True:
        c = min(QUOTA, key=lambda k: (done[k] / QUOTA[k], k))
        pr = pools[c][orders[c][ptr[c] % len(pools[c])]]
        ptr[c] += 1
        n = yield i, c, pr, args.seed * 1_000_003 + i
        done[c] += n
        i += 1


teacher = load(TARGET, TARGET_REV)
for prm in teacher.parameters():
    prm.requires_grad_(False)
student = load(DRAFT, DRAFT_REV)
torch.manual_seed(args.seed)                     # LoRA A initialisation (B = 0): same rule for both methods
student = get_peft_model(student, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                             task_type="CAUSAL_LM"))
for n_, prm in student.named_parameters():
    if prm.requires_grad:
        prm.data = prm.data.float()              # FP32 adapters (backbone stays FP16)
trainable = [prm for prm in student.parameters() if prm.requires_grad]
opt = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda")
cost = {"teacher_generation_s": 0.0, "teacher_scoring_s": 0.0, "student_rollout_s": 0.0, "student_fwd_bwd_s": 0.0,
        "optimizer_s": 0.0}


def sync():
    torch.cuda.synchronize()


def generate(model, ids, cap, seed):
    gen = torch.Generator(device=dev).manual_seed(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out, st = target_only_generate(HFEngine(model), torch.tensor([ids], device=dev), cap, gen, eos=EOS)
    reason = "eos" if st["eos"] else "cap"
    return out, reason


# ---------------------------------------------------------------- off-policy: frozen teacher collection
coll_f = OUT / "teacher_collection.jsonl"
if args.method == "offpolicy" and not coll_f.exists():
    print("generating the frozen teacher-response collection", flush=True)
    sch, rows = schedule(), []
    i, c, pr, rs = next(sch)
    budget = dict(QUOTA)
    while True:
        ids = chat_ids(tok, pr["text"])
        sync(); t0 = time.perf_counter()
        resp, reason = generate(teacher, ids, CAP[c], rs)
        sync(); cost["teacher_generation_s"] += time.perf_counter() - t0
        rows.append({"i": i, "category": c, "source_id": pr["id"], "rollout_seed": rs, "prompt_ids": ids,
                     "response_ids": resp, "termination": reason})
        budget[c] -= len(resp)
        if all(v <= 0 for v in budget.values()):
            break
        try:
            i, c, pr, rs = sch.send(len(resp))
        except StopIteration:
            break
    with open(coll_f, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    save_json({"teacher_generation_s": cost["teacher_generation_s"], "responses": len(rows),
               "tokens": sum(len(r["response_ids"]) for r in rows)}, OUT / "teacher_collection_cost.json")
    print(f"collection: {len(rows)} responses, {sum(len(r['response_ids']) for r in rows)} tokens, "
          f"{cost['teacher_generation_s']:.0f}s", flush=True)
collection = [json.loads(l) for l in open(coll_f)] if args.method == "offpolicy" else None

# ---------------------------------------------------------------- resume
ck = OUT / "checkpoint.pt"
state = {"pres": 0, "updates": 0, "scored": {"math": 0, "short": 0}, "acc_tokens": 0, "skipped": 0}
pres_log, upd_log = [], []
if ck.exists():
    s = torch.load(ck, weights_only=False)
    set_peft_model_state_dict(student, s["adapter"])
    opt.load_state_dict(s["opt"]); scaler.load_state_dict(s["scaler"])
    state, cost, pres_log, upd_log = s["state"], s["cost"], s["pres_log"], s["upd_log"]
    for q, g in zip(trainable, s["grads"]):
        q.grad = g
    print(f"resumed at presentation {state['pres']}, update {state['updates']}", flush=True)


def save_ck():
    torch.save({"adapter": get_peft_model_state_dict(student), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "grads": [None if q.grad is None else q.grad.clone() for q in trainable],
                "state": state, "cost": cost, "pres_log": pres_log, "upd_log": upd_log, "cur_loss": cur_loss}, ck)


def teacher_logprobs(ids, n_score):
    """Teacher log-probabilities at the first n_score response positions (no gradient)."""
    x = torch.tensor([ids], device=dev)
    P = len(ids) - n_score
    sync(); t0 = time.perf_counter()
    with torch.no_grad():
        lp = F.log_softmax(teacher(input_ids=x).logits[0, P - 1:P - 1 + n_score].float(), -1)
    sync(); cost["teacher_scoring_s"] += time.perf_counter() - t0
    return lp


def student_kl(ids, n_score, lp_t, lo, hi):
    """Forward KL(p_T || q_student) at response positions lo..hi-1 (fresh graph each call, so a response
    split across an optimizer update uses the updated adapter for its second part)."""
    x = torch.tensor([ids], device=dev)
    P = len(ids) - n_score
    with torch.autocast("cuda", dtype=torch.float16):
        sl = student(input_ids=x).logits[0, P - 1 + lo:P - 1 + hi]
    lq = F.log_softmax(sl.float(), -1)
    lp = lp_t[lo:hi]
    return (lp.exp() * (lp - lq)).sum(-1)


def optimizer_step(n_tokens):
    sync(); t0 = time.perf_counter()
    scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    before = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    skipped = scaler.get_scale() < before
    opt.zero_grad(set_to_none=True)
    sync(); cost["optimizer_s"] += time.perf_counter() - t0
    state["updates"] += 1
    state["skipped"] += int(skipped)
    return float(gn), skipped


# ---------------------------------------------------------------- training loop
student.train()
sch = schedule()
item = next(sch)
for _ in range(state["pres"]):                    # fast-forward the deterministic schedule on resume
    item = sch.send(pres_log[_]["n_scored"] if _ < len(pres_log) else 0)
cur_loss = s["cur_loss"] if ck.exists() else 0.0
t_start, want_save = time.time(), False
while any(state["scored"][c] < QUOTA[c] for c in QUOTA):
    i, c, pr, rs = item
    if args.method == "offpolicy":
        rec = collection[i % len(collection)]
        assert rec["source_id"] == pr["id"], "collection and schedule out of sync"
        ids_p, resp, reason = rec["prompt_ids"], rec["response_ids"], rec["termination"]
    else:
        ids_p = chat_ids(tok, pr["text"])
        student.eval()
        sync(); t0 = time.perf_counter()
        resp, reason = generate(student, ids_p, CAP[c], rs)
        sync(); cost["student_rollout_s"] += time.perf_counter() - t0
        student.train()
    remaining = QUOTA[c] - state["scored"][c]
    n_score = min(len(resp), remaining)
    if n_score < len(resp):
        reason += "+quota_truncated"
    if n_score > 0:
        seq = ids_p + resp[:n_score]
        lp_t = teacher_logprobs(seq, n_score)
        sync(); t0 = time.perf_counter()
        pos = 0
        while pos < n_score:                       # split across update boundaries
            room = UPDATE_TOKENS - state["acc_tokens"]
            take = min(room, n_score - pos)
            total_left = sum(QUOTA[k] - state["scored"][k] for k in QUOTA) - pos
            denom = UPDATE_TOKENS if total_left >= room else state["acc_tokens"] + total_left
            part = student_kl(seq, n_score, lp_t, pos, pos + take).sum() / denom
            scaler.scale(part).backward()
            cur_loss += float(part.detach())
            state["acc_tokens"] += take
            pos += take
            if state["acc_tokens"] == UPDATE_TOKENS or total_left - take == 0:
                gn, sk = optimizer_step(state["acc_tokens"])
                upd_log.append({"update": state["updates"], "tokens": state["acc_tokens"], "kl": cur_loss,
                                "grad_norm": gn, "skipped": sk, "lr": 3e-4, "scale": scaler.get_scale(),
                                "elapsed_s": time.time() - t_start})
                state["acc_tokens"], cur_loss = 0, 0.0
                if state["updates"] % 20 == 0:
                    want_save = True
                    print(f"[{args.method} s{args.seed}] update {state['updates']} | KL {upd_log[-1]['kl']:.3f} | "
                          f"math {state['scored']['math'] + (pos if c == 'math' else 0)}/30000 "
                          f"short {state['scored']['short'] + (pos if c == 'short' else 0)}/10000 | "
                          f"skipped {state['skipped']} | {time.time() - t_start:.0f}s", flush=True)
        sync(); cost["student_fwd_bwd_s"] += time.perf_counter() - t0
        del lp_t
    state["scored"][c] += n_score
    pres_log.append({"i": i, "category": c, "source_id": pr["id"], "rollout_seed": rs, "response_ids": resp,
                     "termination": reason, "n_scored": n_score, "response_len": len(resp)})
    state["pres"] += 1
    item = sch.send(n_score)
    if want_save:                                   # checkpoint between responses (keeps accumulated grads)
        save_ck()
        want_save = False

# ---------------------------------------------------------------- save
student.save_pretrained(OUT / "adapter")
with open(OUT / "presentations.jsonl", "w") as f:
    for r in pres_log:
        f.write(json.dumps(r) + "\n")
pd.DataFrame(upd_log).to_csv(OUT / "updates.csv", index=False)
coll_cost = json.loads((OUT / "teacher_collection_cost.json").read_text()) if args.method == "offpolicy" else {}
pl = pd.DataFrame(pres_log)
save_json({"method": args.method, "seed": args.seed, "scored_tokens": state["scored"], "updates": state["updates"],
           "skipped_updates": state["skipped"], "presentations": len(pres_log),
           "presentations_by_category": pl.category.value_counts().to_dict(),
           "mean_response_len_by_category": pl.groupby("category").response_len.mean().to_dict(),
           "termination_counts": pl.termination.value_counts().to_dict(),
           "cost_s": {**cost, **({"teacher_generation_s": coll_cost.get("teacher_generation_s", 0)} if coll_cost else {})},
           "train_wall_s": time.time() - t_start, "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
           "trainable_params": sum(p_.numel() for p_ in trainable)}, OUT / "summary.json")
ck.unlink(missing_ok=True)
print(f"[{args.method} s{args.seed}] done: {state['updates']} updates, scored {state['scored']}, "
      f"skipped {state['skipped']}", flush=True)
