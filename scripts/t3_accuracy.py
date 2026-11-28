"""Task 3 step 7 (part 1): standalone answer quality of the target, the original draft and each merged student.

  python scripts/t3_accuracy.py --model original
  python scripts/t3_accuracy.py --model target
  python scripts/t3_accuracy.py --model /kaggle/working/outputs/drafts/offpolicy_s6304
Each model alone generates (our plain sampler, temperature 1, top-p 1, EOS stops) for the 120 held-out
prompts with answer-appropriate caps: 512 new tokens for GSM8K, 32 for short-form extraction.
Two paired repetitions: prompt i of repetition k uses the same seed for every model.
Mathematics: the final answer is taken from \\boxed{...}, else '####', else 'answer is ...', else the
last number in the response; flagged unextractable if none, truncated if the cap was hit before EOS.
Extraction: exact = normalised response equals the gold answer; contains = gold answer appears as a
whole word; compliant = single line with no words beyond the answer (format compliance).
Speculative generation follows the target's distribution, so it is never used here.
Writes results/task3/accuracy/<model>/rep<k>.csv (responses included).
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.common.utils import RESULTS_DIR, SEED
from src.task3.data import DRAFT, DRAFT_REV, TARGET, TARGET_REV, chat_ids, eos_ids, eval_prompts, load, tokenizer
from src.task3.specdec import _sync, HFEngine, target_only_generate

CAP = {"math": 512, "familiar": 32, "shifted": 32}
p = argparse.ArgumentParser()
p.add_argument("--model", required=True, help="original | target | merged draft folder")
p.add_argument("--reps", type=int, default=2)
p.add_argument("--limit", type=int, help="smoke test only")
args = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = tokenizer()
EOS = eos_ids(tok)
if args.model == "original":
    model, name = load(DRAFT, DRAFT_REV), "original"
elif args.model == "target":
    model, name = load(TARGET, TARGET_REV), "target"
else:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()
    name = Path(args.model).name
OUT = RESULTS_DIR / "task3" / "accuracy" / name
OUT.mkdir(parents=True, exist_ok=True)
NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def to_float(s):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def math_answer(text):
    for pat in [r"\\boxed\{([^{}]*)\}", r"####\s*(" + NUM + ")", r"answer is[:\s]*\**\s*(" + NUM + ")"]:
        m = re.findall(pat, text, flags=re.IGNORECASE)
        if m:
            nums = re.findall(NUM, m[-1])
            if nums:
                return nums[-1]
    nums = re.findall(NUM, text)
    return nums[-1] if nums else None


def norm(s):
    return re.sub(r"\s+", " ", s.strip().strip("`'\"*").rstrip(".").strip().lower())


def score(pr, text):
    if pr["domain"] == "math":
        a = math_answer(text)
        v, gold = (to_float(a) if a else None), to_float(pr["answer"])
        return {"extracted": a, "unextractable": a is None, "correct": v is not None and gold is not None and abs(v - gold) < 1e-6}
    r, g = norm(text), norm(pr["answer"])
    return {"extracted": r, "correct": r == g,
            "contains": re.search(r"\b" + re.escape(g) + r"\b", text.lower()) is not None,
            "compliant": "\n" not in text.strip() and len(r.split()) <= len(g.split())}


prompts = eval_prompts()[:args.limit] if args.limit else eval_prompts()
for rep in range(args.reps):
    rows, t_start = [], time.time()
    for i, pr in enumerate(prompts):
        ids = torch.tensor([chat_ids(tok, pr["text"])], device=dev)
        seed = SEED * 1000 + 500 + 100 * rep + i
        gen = torch.Generator(device=dev).manual_seed(seed)
        _sync(dev); t0 = time.perf_counter()
        out, st = target_only_generate(HFEngine(model), ids, CAP[pr["domain"]], gen, eos=EOS)
        _sync(dev)
        text = tok.decode(out, skip_special_tokens=True)
        rows.append({"model": name, "rep": rep, "prompt_id": pr["id"], "domain": pr["domain"], "seed": seed,
                     "n_out": len(out), "eos": st["eos"], "truncated": not st["eos"], "seconds": time.perf_counter() - t0,
                     "gold": pr["answer"], **score(pr, text), "response": text})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"rep{rep}.csv", index=False)
    m, x = df[df.domain == "math"], df[df.domain != "math"]
    msg = f"[{name} rep{rep}] {time.time() - t_start:.0f}s | accuracy " + \
          " ".join(f"{d}:{v.correct.mean():.3f}" for d, v in df.groupby("domain"))
    if len(m):
        msg += f" | math truncated {m.truncated.mean():.3f}, unextractable {m.unextractable.astype(bool).mean():.3f}"
    if len(x):
        msg += f" | extraction compliant {x.compliant.astype(bool).mean():.3f}"
    print(msg, flush=True)
print("Done.", flush=True)
