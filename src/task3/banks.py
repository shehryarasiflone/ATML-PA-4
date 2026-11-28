"""Task 3 prefix banks: reading the supplied banks, the shared position-selection rule, and the
nested-prefix grouping used to score many positions of one response with a single forward pass."""
import gzip
import json
import random
from pathlib import Path

from src.common.data import data_root, records
from src.common.utils import RESULTS_DIR

SUPPLIED = ("teacher", "original_draft")
BINS = [(1, 16, "1-16"), (17, 32, "17-32"), (33, 64, "33-64"), (65, 192, "65-192")]
BANK_DIR = RESULTS_DIR / "task3" / "banks"


def pos_bin(pos):
    for lo, hi, name in BINS:
        if lo <= pos <= hi:
            return name
    raise ValueError(pos)


def select_positions(pid, bank, n_tokens):
    """Same rule as the course setup script: 8 / 8 / 16 positions from [1,16] / [17,32] / [33,64]
    and 16 from [65,192] (fewer if the response is shorter), with a bank-specific seeded RNG."""
    rng = random.Random(f"PA4-banks-v1:6304:{pid}:{bank}")
    out = []
    for lo, hi, n in [(1, 16, 8), (17, 32, 8), (33, 64, 16), (65, 192, 16)]:
        cand = list(range(lo, min(hi, n_tokens) + 1))
        out.extend(sorted(rng.sample(cand, min(n, len(cand)))))
    return out


def supplied_path():
    return data_root() / "prefix_banks" / "prefix_banks_locked.jsonl.gz"


def load_bank(name):
    """Records of one bank: the two supplied banks come from the course file, the others from
    results/task3/banks/<name>.jsonl.gz written by scripts/t3_banks.py."""
    if name in SUPPLIED:
        p = supplied_path()
        if not p.exists() and not p.with_suffix("").exists():
            raise FileNotFoundError(f"supplied prefix banks not found at {p} (run setup_data.py --prefix-banks "
                                    "and include data/prefix_banks in the Kaggle dataset)")
        return [r for r in records(p) if r["bank"] == name]
    return records(BANK_DIR / f"{name}.jsonl.gz")


def save_bank(name, rows):
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(BANK_DIR / f"{name}.jsonl.gz", "wt", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def grouped(rows):
    """{prompt_id: (prompt_text, longest_prefix, [records])}. All prefixes of one prompt in one bank
    are prefixes of the same response, so one forward pass over prompt + longest prefix gives the
    next-token distribution at every saved position (checked here)."""
    by = {}
    for r in rows:
        by.setdefault(r["prompt_id"], []).append(r)
    out = {}
    for pid, rs in by.items():
        rs.sort(key=lambda r: r["position"])
        longest = max((r["response_prefix_token_ids"] for r in rs), key=len)
        for r in rs:
            pre = r["response_prefix_token_ids"]
            assert len(pre) == r["position"] - 1, (pid, r["position"], len(pre))
            assert longest[:len(pre)] == pre, f"prefixes of {pid} are not nested"
        out[pid] = (rs[0]["prompt"], longest, rs)
    return out
