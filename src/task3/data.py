"""Task 3 prompts and model loading (shared chat format: default Qwen system prompt, one user turn,
generation prompt appended; no thinking mode exists for Qwen2.5)."""
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.data import data_root, records

TARGET, TARGET_REV = "Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DRAFT, DRAFT_REV = "Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"


def load(name, rev, device="cuda"):
    return AutoModelForCausalLM.from_pretrained(name, revision=rev, dtype=torch.float16,
                                                attn_implementation="sdpa").to(device).eval()


def tokenizer():
    return AutoTokenizer.from_pretrained(TARGET, revision=TARGET_REV)


def eos_ids(tok):
    return {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}


def chat_ids(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)


def eval_prompts():
    """The 120 held-out prompts: 60 GSM8K test (domain 'math') + 60 short-form (familiar / shifted)."""
    out = []
    for r in sorted(records(data_root() / "public_locked" / "gsm8k_test_locked.jsonl.gz"), key=lambda r: r["order"]):
        out.append({"id": f"GSM8K-{r['order']:04d}", "text": r["example"]["question"], "domain": "math",
                    "answer": r["example"]["answer"].split("####")[-1].strip().replace(",", "")})
    for r in records(data_root() / "short_form" / "short_test.jsonl"):
        out.append({"id": r["id"], "text": r["input"], "domain": r["split"], "family": r["family"], "answer": r["answer"]})
    return out


def train_prompts():
    """500 GSM8K train questions + 300 short-form training instructions."""
    math = [{"id": f"GSM8K-train-{r['order']:04d}", "text": r["example"]["question"], "domain": "math"}
            for r in sorted(records(data_root() / "public_locked" / "gsm8k_train_locked.jsonl.gz"), key=lambda r: r["order"])]
    short = [{"id": r["id"], "text": r["input"], "domain": "short", "family": r["family"]}
             for r in records(data_root() / "short_form" / "short_train.jsonl")]
    return math, short
