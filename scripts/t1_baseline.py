"""Task 1, step 1: unchanged FP16 baseline and the shared measurement protocol.

Run:  python scripts/t1_baseline.py
Writes results/task1/conditions/fp16_baseline/ and results/task1/baseline_checks.json
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.common.data import data_root, records
from src.common.utils import RESULTS_DIR, SEED, save_json, set_seed
from src.task1.evaluate import T1_DIR, evaluate_condition, load_calibration, load_hellaswag, load_windows
from src.task1.model_utils import ELIGIBLE, eligible_linears, layer_and_type, load_base, load_tokenizer, param_accounting

set_seed(SEED)
root = data_root()
print("data:", root, flush=True)
(RESULTS_DIR / "env").mkdir(parents=True, exist_ok=True)
shutil.copy(root / "local_source_revisions.json", RESULTS_DIR / "env" / "data_source_revisions.json")

tok = load_tokenizer()
model = load_base()
checks = {}

# 1. Eligible matrices and tied weights ---------------------------------------
elig = eligible_linears(model)
L = model.config.num_hidden_layers
per = {}
for n, m in elig:
    li, t = layer_and_type(n)
    per.setdefault(t, set()).add(li)
assert len(elig) == 7 * L, len(elig)
assert all(per[t] == set(range(L)) for t in ELIGIBLE), "every block must have all 7 projections"
acc = param_accounting(model)
acc["lm_head_tied_to_embeddings"] = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
acc["lm_head_in_eligible"] = any(m.weight.data_ptr() == model.lm_head.weight.data_ptr() for _, m in elig)
assert acc["lm_head_tied_to_embeddings"] and not acc["lm_head_in_eligible"]
assert acc["eligible_nonzero_fraction"] > 0.999
acc["shapes"] = {t: list(dict(elig)[f"model.layers.0.{'self_attn' if t in ELIGIBLE[:4] else 'mlp'}.{t}"].weight.shape)
                 for t in ELIGIBLE}
acc["fp16_dense_bytes_eligible"] = 2 * acc["eligible_weights"]
acc["fp16_dense_bytes_total"] = 2 * acc["total_params_tied_once"]
checks["params"] = acc
print("eligible:", acc["eligible_matrices"], "matrices,", f"{acc['eligible_weights']:,}", "weights | other:",
      f"{acc['other_params_uncompressed']:,}", "| shapes:", acc["shapes"], flush=True)

# 2. Data checks ---------------------------------------------------------------
cal, cal_rows = load_calibration("calibration")
dev, dev_rows = load_calibration("development")
checks["calibration"] = {"shape": list(cal.shape), "dev_shape": list(dev.shape),
                         "overlap_rows": len(set(cal_rows) & set(dev_rows))}
for suite in ["wikitext2", "c4val"]:
    w = load_windows(suite)
    checks[suite] = {"windows": len(w), "scored_tokens": sum(x["scored_tokens"] for x in w),
                     "max_len": max(len(x["token_ids"]) for x in w)}
assert checks["calibration"]["overlap_rows"] == 0
print("data checks:", {k: checks[k] for k in ["calibration", "wikitext2", "c4val"]}, flush=True)

# 3. Baseline evaluation (full HellaSwag) --------------------------------------
hs = load_hellaswag(tok)
checks["hellaswag_example0"] = {"ctx": tok.decode(hs[0]["ctx_ids"]),
                                "cont0": tok.decode(hs[0]["cont_ids"][0]), "gold": hs[0]["gold"]}
summary = evaluate_condition(model, tok, "fp16_baseline", hs="full", hs_cache=hs,
                             extra={"params": acc, "storage_bytes_total": acc["fp16_dense_bytes_total"]})

# 4. Diagnostic subset = first 512 of the same run (sanity: must match a direct run) -----
import pandas as pd  # noqa: E402
df = pd.read_csv(T1_DIR / "conditions" / "fp16_baseline" / "hellaswag_full.csv")
checks["hellaswag_diag_from_full"] = {"acc": float(df.correct[:512].mean()), "acc_norm": float(df.correct_norm[:512].mean())}
save_json(checks, T1_DIR / "baseline_checks.json")
print("diag subset (first 512):", checks["hellaswag_diag_from_full"])
print("Done. Saved results/task1/conditions/fp16_baseline/ and results/task1/baseline_checks.json")
