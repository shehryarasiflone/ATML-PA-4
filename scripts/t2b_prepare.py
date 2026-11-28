"""Task 2 Part B: verify the supplied MQAR generator and build the cached training streams.

  python scripts/t2b_prepare.py
Checks: every frozen test/development record passes the generator's validate(); the generator
reproduces a sample of frozen test records exactly; regime distance limits; no value IDs in filler
positions; training indices never collide with the test/development records (different split key).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.common.data import data_root, records
from src.common.utils import RESULTS_DIR, save_json
from src.task2.mqar import generator_module, train_stream

g = generator_module()
out = {}
for split in ["test", "development"]:
    rows = records(data_root() / "mqar" / f"mqar_{split}.jsonl.gz")
    for r in rows:
        g.validate(r)
    out[f"{split}_records_validated"] = len(rows)
    filler_values = 0
    for r in rows:
        qs = set(r["query_positions"])
        K = r["K"]
        for p, tkn in enumerate(r["tokens"]):
            if p >= 2 * K and p not in qs and 128 <= tkn < 192:
                filler_values += 1
    out[f"{split}_value_ids_in_filler"] = filler_values
    d = {reg: [x for r in rows if r["regime"] == reg for x in r["distances"]] for reg in ["near", "far", "mixed"]}
    out[f"{split}_distance_ranges"] = {reg: [min(v), max(v), float(np.mean(v))] for reg, v in d.items()}
    far = d["far"]
    out[f"{split}_far_offsets_not_div4_share"] = float(np.mean([x % 4 != 0 for x in far]))
test = records(data_root() / "mqar" / "mqar_test.jsonl.gz")
mism = 0
for r in test[::64]:
    i = int(r["id"].split("-")[-1])
    if g.make_example("test", r["K"], r["regime"], i) != r:
        mism += 1
out["regenerated_test_sample"] = len(test[::64])
out["regenerated_mismatches"] = mism
print(out, flush=True)
assert mism == 0 and out["test_value_ids_in_filler"] == 0
save_json(out, RESULTS_DIR / "task2" / "mqar" / "generator_checks.json")

t0 = time.time()
for seed in (6304, 6305, 6306):
    for K in (4, 8, 16):
        for reg in ("mixed", "near"):
            a = train_stream(K, reg, seed, workers=4)
            print(f"stream K={K} {reg} seed={seed}: {a['tokens'].shape} ({time.time() - t0:.0f}s)", flush=True)
print("Done.", flush=True)
