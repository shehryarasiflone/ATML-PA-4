"""Locate the prepared PA4 data folder (course data repo after setup_data.py) and read JSONL."""
import gzip
import json
import os
from pathlib import Path

from src.common.utils import REPO_ROOT


def data_root() -> Path:
    """Return the `data/` folder that holds public_locked/, mqar/, kv_retrieval/, ...

    Search order: $PA4_DATA, Kaggle inputs (the uploaded dataset), a sibling clone.
    """
    cands = []
    if os.environ.get("PA4_DATA"):
        cands.append(Path(os.environ["PA4_DATA"]))
    kin = Path("/kaggle/input")
    if kin.is_dir():
        cands += sorted(kin.glob("*/data")) + sorted(kin.glob("*/*/data")) + sorted(kin.glob("*/*/*/data"))
    cands += [Path("/kaggle/working/atml-pa4-datasets/data"), REPO_ROOT.parent / "atml-pa4-datasets" / "data"]
    for c in cands:
        if (c / "public_locked").is_dir() or (c / "mqar").is_dir():
            return c
    raise FileNotFoundError("PA4 data not found. Attach the pa4-dataset Kaggle dataset as an input "
                            "or set PA4_DATA to the data/ folder.")


def records(path):
    path = Path(path)
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
