"""Locate the prepared PA4 data folder (course data repo after setup_data.py) and read JSONL."""
import gzip
import json
import os
from pathlib import Path

from src.common.utils import REPO_ROOT

KAGGLE_INPUT = "/kaggle/input"


def data_root() -> Path:
    """Return the `data/` folder that holds public_locked/, mqar/, kv_retrieval/, ...

    Search order: $PA4_DATA, Kaggle inputs (the uploaded dataset), a sibling clone.
    """
    cands = []
    if os.environ.get("PA4_DATA"):
        cands.append(Path(os.environ["PA4_DATA"]))
    # Kaggle inputs: search any depth (mount paths differ, e.g. /kaggle/input/datasets/<user>/<name>/...)
    for top in [KAGGLE_INPUT]:
        for dirpath, dirnames, _ in os.walk(top):
            if "public_locked" in dirnames or "mqar" in dirnames:
                cands.append(Path(dirpath))
            if dirpath.count(os.sep) - top.count(os.sep) >= 6:
                dirnames[:] = []          # do not walk deeper than 6 levels
    cands += [Path("/kaggle/working/atml-pa4-datasets/data"), REPO_ROOT.parent / "atml-pa4-datasets" / "data"]
    for c in cands:
        if (c / "public_locked").is_dir() or (c / "mqar").is_dir():
            return c
    raise FileNotFoundError("PA4 data not found. Attach the pa4-dataset Kaggle dataset as an input "
                            "or set PA4_DATA to the data/ folder.")


def records(path):
    """Read JSONL or JSONL.GZ."""
    path = Path(path)
    if not path.exists() and path.suffix == ".gz" and path.with_suffix("").exists():
        path = path.with_suffix("")
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]