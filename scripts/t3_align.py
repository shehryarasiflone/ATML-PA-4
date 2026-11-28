"""Task 3 step 5 (part 2): distribution alignment of every draft with the target on common prefixes.

  python scripts/t3_align.py --banks teacher exact_spec_original_gamma4 student_offpolicy_s6304
Drafts: the original 0.5B and the four merged students in /kaggle/working/outputs/drafts/ (all are
evaluated on every bank, so each student is also scored on the other students' frozen rollouts).
At every saved prefix h (prompt in the shared chat format + response_prefix_token_ids) we compute,
over the full 151,936-entry vocabulary in FP32 from FP16 model logits, with p = target and q = draft:
  forward KL(p||q), reverse KL(q||p), TV(p,q), expected one-token acceptance 1 - TV,
  greedy agreement [argmax p == argmax q], and the target entropy H(p).
Each record is one response position, so averages over records are token-weighted.
Writes results/task3/align/<bank>.csv.gz (one row per prefix, one column per draft and metric,
plus the decoded tail of the prefix and the top tokens, for prefix-level examples).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from src.common.utils import RESULTS_DIR, work_dir
from src.task3.banks import grouped, load_bank, pos_bin
from src.task3.data import DRAFT, DRAFT_REV, TARGET, TARGET_REV, chat_ids, eval_prompts, load, tokenizer

RUNS = ["offpolicy_s6304", "offpolicy_s6305", "onpolicy_s6304", "onpolicy_s6305"]
p = argparse.ArgumentParser()
p.add_argument("--banks", nargs="+", required=True)
p.add_argument("--drafts-dir", default=str(work_dir() / "drafts"))
args = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = tokenizer()
domain = {pr["id"]: pr["domain"] for pr in eval_prompts()}          # math / familiar / shifted
OUT = RESULTS_DIR / "task3" / "align"
OUT.mkdir(parents=True, exist_ok=True)

tm = load(TARGET, TARGET_REV)
drafts = {"original": load(DRAFT, DRAFT_REV)}
for r in RUNS:
    d = Path(args.drafts_dir) / r
    if not (d / "config.json").exists():
        raise FileNotFoundError(f"merged draft {d} missing: run scripts/t3_merge.py first")
    drafts[r] = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()


for bank in args.banks:
    t0 = time.time()
    groups = grouped(load_bank(bank))
    table = []
    for pid, (prompt, longest, recs) in groups.items():
        pids = chat_ids(tok, prompt)
        ids = torch.tensor([pids + longest], device=dev)
        # the distribution of response token `position` is read at sequence index len(prompt)+position-2
        idx = torch.tensor([len(pids) + r["position"] - 2 for r in recs], device=dev)
        with torch.no_grad():
            lp = F.log_softmax(tm(input_ids=ids).logits[0, idx].float(), -1)
            pt = lp.exp()
            rows = [{"bank": bank, "prompt_id": pid, "domain": domain.get(pid, recs[0].get("domain")),
                     "position": r["position"], "bin": pos_bin(r["position"]), "source_length": r.get("source_length"),
                     "producer": r.get("producing_checkpoint", bank),
                     "context_tail": tok.decode(r["response_prefix_token_ids"][-12:]) if r["position"] > 1 else "",
                     "H_target": float(-(pt[k] * lp[k]).sum())} for k, r in enumerate(recs)]
            top_p = pt.max(-1)
            for k in range(len(recs)):
                rows[k]["target_top"] = tok.decode([int(top_p.indices[k])])
                rows[k]["target_top_p"] = float(top_p.values[k])
            for name, dm in drafts.items():
                lq = F.log_softmax(dm(input_ids=ids).logits[0, idx].float(), -1)
                q = lq.exp()
                fkl = (pt * (lp - lq)).sum(-1)
                rkl = (q * (lq - lp)).sum(-1)
                tv = 0.5 * (pt - q).abs().sum(-1)
                top_q = q.max(-1)
                for k in range(len(recs)):
                    rows[k].update({f"fkl__{name}": float(fkl[k]), f"rkl__{name}": float(rkl[k]),
                                    f"tv__{name}": float(tv[k]), f"acc__{name}": 1.0 - float(tv[k]),
                                    f"greedy__{name}": int(top_q.indices[k] == top_p.indices[k]),
                                    f"top__{name}": tok.decode([int(top_q.indices[k])]),
                                    f"top_q__{name}": float(top_q.values[k])})
        table += rows
    df = pd.DataFrame(table)
    df.to_csv(OUT / f"{bank}.csv.gz", index=False)
    summ = {n: f"TV {df[f'tv__{n}'].mean():.3f} fKL {df[f'fkl__{n}'].mean():.3f} greedy {df[f'greedy__{n}'].mean():.3f}"
            for n in drafts}
    print(f"[{bank}] {len(df)} prefixes, {len(groups)} prompts, H_target {df.H_target.mean():.3f}, "
          f"{time.time() - t0:.0f}s", flush=True)
    for n, s in summ.items():
        print(f"    {n:16s} {s}", flush=True)
print("Done.", flush=True)
