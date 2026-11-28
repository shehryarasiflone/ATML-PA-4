"""Task 3 step 1: validate the exact speculative sampler.

  python scripts/t3_validate.py
1. Toy 3-symbol Markov target/draft: 100,000 fixed-seed sequences of length 3 for gamma in {1, 2, 3, 4}
   (gamma 4 > 3 exercises final-token truncation), a variant with an EOS symbol (variable-length
   sequences), and the deliberately wrong sampler (residual replaced by p). Pearson chi-square against
   the exact joint sequence probabilities (all expected counts >= 10).
2. Edge cases: one-token rounds, draft == target (every proposal accepted), EOS stop, length cap.
3. Real models: tokenizer / chat-template agreement on all 920 prompts; split-prefill equivalence;
   cache rollback consistency; target-as-its-own-draft acceptance.
Writes results/task3/validation.json and results/task3/toy_joint.csv.
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from scipy.stats import chisquare

from src.common.utils import RESULTS_DIR, SEED, save_json
from src.task3.data import DRAFT, DRAFT_REV, TARGET, TARGET_REV, chat_ids, eos_ids, eval_prompts, load, tokenizer, train_prompts
from src.task3.specdec import HFEngine, MarkovEngine, speculative_generate

OUT = RESULTS_DIR / "task3"
OUT.mkdir(parents=True, exist_ok=True)
res = {}
N = 100_000

# ---------------------------------------------------------------- 1. toy Markov chain
P = torch.tensor([[0.50, 0.30, 0.20], [0.15, 0.25, 0.60], [0.35, 0.45, 0.20]])   # target p(b | a)
Q = torch.tensor([[0.20, 0.30, 0.50], [0.40, 0.40, 0.20], [0.10, 0.30, 0.60]])   # draft  q(b | a)
START = 0


def exact_joint(eos=None, length=3):
    probs = {}
    def rec(seq, last, pr):
        if len(seq) == length or (eos is not None and seq and seq[-1] == eos):
            probs[tuple(seq)] = probs.get(tuple(seq), 0) + pr
            return
        for b in range(3):
            rec(seq + [b], b, pr * float(P[last, b]))
    rec([], START, 1.0)
    return probs


def run(gamma, wrong=False, eos=None, seed=SEED):
    gen = torch.Generator().manual_seed(seed)
    tgt, dft = MarkovEngine(P, P[START]), MarkovEngine(Q, Q[START])
    counts, acc, prop = {}, 0, 0
    for _ in range(N):
        out, st = speculative_generate(tgt, dft, [START], gamma, 3, gen, eos=() if eos is None else (eos,),
                                       wrong_residual=wrong)
        counts[tuple(out)] = counts.get(tuple(out), 0) + 1
        acc += st["accepted"]; prop += st["proposed"]
    return counts, acc / prop


rows = []
for name, gamma, wrong, eos in [("gamma1", 1, False, None), ("gamma2", 2, False, None), ("gamma3", 3, False, None),
                                ("gamma4_truncation", 4, False, None), ("gamma2_eos", 2, False, 2),
                                ("gamma2_WRONG_residual_p", 2, True, None)]:
    ex = exact_joint(eos)
    cnt, acc = run(gamma, wrong, eos)
    keys = sorted(ex)
    obs = [cnt.get(k, 0) for k in keys]
    tot = sum(ex[k] for k in keys)
    exp = [N * ex[k] / tot for k in keys]
    assert min(exp) >= 10 and sum(obs) == N and set(cnt) <= set(ex)
    chi = chisquare(obs, exp)
    res[f"toy_{name}"] = {"chi2": float(chi.statistic), "dof": len(keys) - 1, "p_value": float(chi.pvalue),
                         "rejected_at_0.01": bool(chi.pvalue < 0.01), "acceptance_rate": acc,
                         "max_abs_freq_minus_prob": max(abs(o / N - e / N) for o, e in zip(obs, exp))}
    for k, o, e in zip(keys, obs, exp):
        rows.append({"test": name, "sequence": "".join(map(str, k)), "exact_prob": e / N, "empirical": o / N, "count": o})
    print(name, res[f"toy_{name}"], flush=True)
pd.DataFrame(rows).to_csv(OUT / "toy_joint.csv", index=False)
# Any correct-sampler test rejected at p < 0.01 is replicated with other seeds and 4x the samples
# (5 correct-sampler tests at alpha = 0.01 give ~5% chance of at least one false rejection).
for name, gamma, eos in [("gamma1", 1, None), ("gamma2", 2, None), ("gamma3", 3, None),
                         ("gamma4_truncation", 4, None), ("gamma2_eos", 2, 2)]:
    if res[f"toy_{name}"]["rejected_at_0.01"]:
        ex = exact_joint(eos)
        keys = sorted(ex)
        tot = sum(ex.values())
        reps = []
        for n_s, seed in [(N, 1), (N, 2), (N, 3), (4 * N, 4)]:
            N_save, N = N, n_s
            cnt, _ = run(gamma, False, eos, seed)
            N = N_save
            reps.append({"n": n_s, "seed": seed, "p_value": float(chisquare(
                [cnt.get(k, 0) for k in keys], [n_s * ex[k] / tot for k in keys]).pvalue)})
        res[f"toy_{name}"]["replications"] = reps
        print(name, "replications:", reps, flush=True)
        assert all(r["p_value"] >= 0.01 for r in reps), f"{name} fails on replication: sampler bug"
assert res["toy_gamma2_WRONG_residual_p"]["rejected_at_0.01"]
# 1-step TV identity: E_q[min(1, p/q)] = 1 - TV(p, q) for the first proposal
tv = 0.5 * float((P[START] - Q[START]).abs().sum())
res["toy_first_step_expected_acceptance_1_minus_TV"] = 1 - tv

# ---------------------------------------------------------------- 2. edge cases
gen = torch.Generator().manual_seed(SEED)
same = [speculative_generate(MarkovEngine(P, P[0]), MarkovEngine(P, P[0]), [0], 3, 7, gen)[1] for _ in range(200)]
res["edge_draft_equals_target_acceptance"] = sum(s["accepted"] for s in same) / sum(s["proposed"] for s in same)
out, st = speculative_generate(MarkovEngine(P, P[0]), MarkovEngine(Q, Q[0]), [0], 1, 5, gen)
res["edge_gamma1_rounds_vs_tokens"] = [st["rounds"], len(out)]
out, st = speculative_generate(MarkovEngine(P, P[0]), MarkovEngine(Q, Q[0]), [0], 4, 6, gen)
res["edge_length_cap_respected"] = len(out) == 6
lens = [len(speculative_generate(MarkovEngine(P, P[0]), MarkovEngine(Q, Q[0]), [0], 3, 50, gen, eos=(2,))[0]) for _ in range(200)]
res["edge_eos_stops_mean_len"] = sum(lens) / len(lens)
print("edge cases:", {k: v for k, v in res.items() if k.startswith("edge")}, flush=True)

# ---------------------------------------------------------------- 3. real models
tok = tokenizer()
from transformers import AutoTokenizer  # noqa: E402
dtok = AutoTokenizer.from_pretrained(DRAFT, revision=DRAFT_REV)
math_tr, short_tr = train_prompts()
allp = eval_prompts() + math_tr + short_tr
res["chat_ids_identical_target_vs_draft"] = all(chat_ids(tok, p["text"]) == chat_ids(dtok, p["text"]) for p in allp)
res["n_prompts_checked"] = len(allp)
tm, dm = load(TARGET, TARGET_REV), load(DRAFT, DRAFT_REV)
res["vocab_rows_target_draft"] = [tm.get_output_embeddings().weight.shape[0], dm.get_output_embeddings().weight.shape[0]]
dev = "cuda"
ids = chat_ids(tok, eval_prompts()[0]["text"])
with torch.no_grad():
    full = torch.softmax(tm(input_ids=torch.tensor([ids], device=dev)).logits[0].float(), -1)
e = HFEngine(tm)
p0 = e.prefill(torch.tensor([ids[:-6]], device=dev))
ps = e.extend(ids[-6:])
res["split_prefill_max_abs_prob_diff"] = float(max((p0 - full[-7]).abs().max(), (ps - full[-6:]).abs().max()))
before = e.extend([100, 200, 300, 400])
e.rollback(e.length - 4)
after = e.extend([100, 200, 300, 400])
res["rollback_reextend_max_abs_prob_diff"] = float((before - after).abs().max())
res["rollback_length_consistent"] = e.length == len(ids) + 4 and e.cache.get_seq_length() == e.length
gen = torch.Generator(device=dev).manual_seed(SEED)
st_all = [speculative_generate(HFEngine(tm), HFEngine(tm), torch.tensor([chat_ids(tok, p["text"])], device=dev), 4, 32, gen,
                               eos=eos_ids(tok))[1] for p in eval_prompts()[:5]]
res["target_as_own_draft_acceptance"] = sum(s["accepted"] for s in st_all) / sum(s["proposed"] for s in st_all)
print("real-model checks:", {k: v for k, v in res.items() if not k.startswith(("toy", "edge"))}, flush=True)
save_json(res, OUT / "validation.json")
print("Done.", flush=True)
