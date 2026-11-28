"""Task 3 step 5 (part 1): generate the prefix banks that are not supplied.

  python scripts/t3_banks.py --spec                                            # exact_spec_original_gamma4
  python scripts/t3_banks.py --student /kaggle/working/outputs/drafts/offpolicy_s6304   # student_offpolicy_s6304
For each of the 120 held-out prompts (same order, chat format and text as the supplied banks) one
response of up to 192 tokens is sampled at temperature 1, top-p 1, stopping at EOS:
  --spec     with our exact speculative sampler (target 1.5B, original 0.5B draft, gamma 4);
             this bank should follow the same prefix law as the supplied teacher bank.
  --student  from a merged trained student on its own (its frozen own-policy rollouts); the
             producing checkpoint path is stored in every record.
  --target   diagnostic: the target alone with our plain sampler (pure temperature 1, top-p 1).
  --hf-teacher default|norep   diagnostic: the supplied teacher bank's own procedure (HF generate,
             same seeds); 'default' keeps the model's generation_config (reports how many responses
             are identical to the supplied ones), 'norep' sets repetition_penalty=1.0.
Positions are then chosen with the same rule as the supplied banks (8/8/16 in 1-64, 16 in 65-192).
Writes results/task3/banks/<bank>.jsonl.gz and <bank>_rollouts.jsonl (full responses, termination,
and for --spec the per-response acceptance statistics).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.task3.banks import BANK_DIR, load_bank, save_bank, select_positions
from src.task3.data import DRAFT, DRAFT_REV, TARGET, TARGET_REV, chat_ids, eos_ids, eval_prompts, load, tokenizer
from src.task3.specdec import _sync, HFEngine, speculative_generate, target_only_generate

p = argparse.ArgumentParser()
g = p.add_mutually_exclusive_group(required=True)
g.add_argument("--spec", action="store_true")
g.add_argument("--student")
g.add_argument("--target", action="store_true")
g.add_argument("--hf-teacher", choices=["default", "norep"])
p.add_argument("--limit", type=int, help="smoke test only")
args = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = tokenizer()
EOS = eos_ids(tok)
MAX_NEW = 192

if args.spec:
    name, offset = "exact_spec_original_gamma4", 456
    tm, dm = load(TARGET, TARGET_REV), load(DRAFT, DRAFT_REV)
    producer = f"speculative(target={TARGET}@{TARGET_REV[:8]}, draft={DRAFT}@{DRAFT_REV[:8]}, gamma=4)"
elif args.target:
    name, offset = "target_plain", 789
    tm = load(TARGET, TARGET_REV)
    producer = f"target-only sampler (ours), {TARGET}@{TARGET_REV[:8]}, T=1, top-p 1"
elif args.hf_teacher:
    name, offset = f"teacher_hf_{args.hf_teacher}", 0          # same seeds as the supplied teacher bank
    tm = load(TARGET, TARGET_REV)
    print("target generation_config:", tm.generation_config.to_dict(), flush=True)
    producer = f"HF generate as in setup_data.py ({args.hf_teacher} generation_config), {TARGET}@{TARGET_REV[:8]}"
else:
    from transformers import AutoModelForCausalLM
    run = Path(args.student).name
    name, offset = f"student_{run}", 1000 + 100 * ["offpolicy_s6304", "offpolicy_s6305",
                                                   "onpolicy_s6304", "onpolicy_s6305"].index(run)
    sm = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()
    producer = str(Path(args.student).resolve())

if (BANK_DIR / f"{name}.jsonl.gz").exists():
    print(name, "already exists; delete it to regenerate", flush=True)
    sys.exit(0)

# same prompts, order and text as the supplied teacher bank
prompts = eval_prompts()
teacher_bank = load_bank("teacher")
supplied = {r["prompt_id"]: r["prompt"] for r in teacher_bank}
longest_supplied = {}
for r in teacher_bank:
    if len(r["response_prefix_token_ids"]) > len(longest_supplied.get(r["prompt_id"], [])):
        longest_supplied[r["prompt_id"]] = r["response_prefix_token_ids"]
for pr in prompts:
    assert supplied.get(pr["id"], pr["text"]) == pr["text"], f"prompt text differs from the supplied bank: {pr['id']}"
if args.limit:
    prompts = prompts[:args.limit]

bank, rolls = [], []
t_start = time.time()
for j, pr in enumerate(prompts):
    ids = torch.tensor([chat_ids(tok, pr["text"])], device=dev)
    seed = 6304 + 10007 * j + offset
    gen = torch.Generator(device=dev).manual_seed(seed)
    _sync(dev); t0 = time.perf_counter()
    if args.spec:
        seq, st = speculative_generate(HFEngine(tm), HFEngine(dm), ids, 4, MAX_NEW, gen, eos=EOS)
        extra = {k: st[k] for k in ["rounds", "proposed", "examined", "accepted", "bonus", "target_passes", "draft_passes"]}
    elif args.hf_teacher:
        # the course setup script's call (generator= replaced by torch.manual_seed, as in our patched copy);
        # 'norep' additionally switches off the repetition penalty stored in the model's generation_config
        torch.manual_seed(seed)
        kw = {"repetition_penalty": 1.0} if args.hf_teacher == "norep" else {}
        with torch.inference_mode():
            seq = tm.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_NEW, do_sample=True,
                              temperature=1.0, top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id, **kw)[0, ids.shape[1]:].tolist()
        st = {"eos": bool(seq) and seq[-1] in EOS}
        ref = longest_supplied.get(pr["id"], [])
        extra = {"matches_supplied_teacher": seq[:len(ref)] == ref}
    else:
        seq, st = target_only_generate(HFEngine(sm if args.student else tm), ids, MAX_NEW, gen, eos=EOS)
        extra = {}
    _sync(dev)
    rolls.append({"prompt_id": pr["id"], "domain": pr["domain"], "seed": seed, "response_ids": seq,
                  "length": len(seq), "termination": "eos" if st["eos"] else "cap",
                  "seconds": time.perf_counter() - t0, "producer": producer, **extra})
    for pos in select_positions(pr["id"], name, len(seq)):
        bank.append({"prompt_id": pr["id"], "domain": pr["domain"], "bank": name, "position": pos, "prompt": pr["text"],
                     "response_prefix_token_ids": seq[:pos - 1], "source_length": len(seq),
                     "source_generation_seed": seed, "producing_checkpoint": producer})
    if (j + 1) % 20 == 0:
        print(f"[{name}] {j + 1}/{len(prompts)} prompts | {len(bank)} prefixes | {time.time() - t_start:.0f}s", flush=True)

save_bank(name, bank)
with open(BANK_DIR / f"{name}_rollouts.jsonl", "w") as f:
    for r in rolls:
        f.write(json.dumps(r) + "\n")
msg = f"[{name}] done: {len(bank)} prefixes, mean length {sum(r['length'] for r in rolls) / len(rolls):.1f}, " \
      f"EOS {sum(r['termination'] == 'eos' for r in rolls)}/{len(rolls)}"
if args.hf_teacher:
    msg += f", identical to the supplied teacher response on {sum(r['matches_supplied_teacher'] for r in rolls)}/{len(rolls)} prompts"
if args.spec:
    msg += f", acceptance {sum(r['accepted'] for r in rolls) / sum(r['examined'] for r in rolls):.3f}"
print(msg, flush=True)
