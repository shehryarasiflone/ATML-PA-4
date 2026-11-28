"""Task 3 step 6 (first half): merge each LoRA adapter into a fresh FP16 0.5B backbone and check it.

  python scripts/t3_merge.py                  # every results/task3/distill/*/adapter
Merging is done in FP32 (base weights upcast, W + (alpha/r) B A), then the merged model is cast to FP16
and saved to /kaggle/working/drafts/<run>/ (large; not part of results/). Check: logits of the merged
FP16 model vs. the unmerged adapter path used in training/rollouts (FP16 backbone + FP32 adapter
under FP16 autocast) on 8 held-out prompts: max |diff|, mean |diff|, top-1 agreement.
Writes results/task3/distill/<run>/merge_check.json.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from src.common.utils import RESULTS_DIR, save_json, work_dir
from src.task3.data import DRAFT, DRAFT_REV, chat_ids, eval_prompts, load, tokenizer

dev = "cuda"
tok = tokenizer()
prompts = [chat_ids(tok, p["text"]) for p in eval_prompts()[::15]][:8]
DR = work_dir() / "drafts"
for ad in sorted((RESULTS_DIR / "task3" / "distill").glob("*/adapter")):
    run = ad.parent.name
    dst = DR / run
    if (ad.parent / "merge_check.json").exists() and (dst / "config.json").exists():
        print(run, "already merged", flush=True)
        continue
    t0 = time.time()
    base32 = AutoModelForCausalLM.from_pretrained(DRAFT, revision=DRAFT_REV, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base32, ad).merge_and_unload().half()
    merged.save_pretrained(dst)
    merge_s = time.time() - t0
    merged = merged.to(dev).eval()
    unmerged = PeftModel.from_pretrained(load(DRAFT, DRAFT_REV), ad).eval()
    for n, p in unmerged.named_parameters():
        if "lora_" in n:
            p.data = p.data.float()
    mx, mean, agree, cnt = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for ids in prompts:
            x = torch.tensor([ids], device=dev)
            a = merged(input_ids=x).logits[0].float()
            with torch.autocast("cuda", dtype=torch.float16):
                b = unmerged(input_ids=x).logits[0].float()
            d = (a - b).abs()
            mx, mean = max(mx, float(d.max())), mean + float(d.mean())
            agree += int((a.argmax(-1) == b.argmax(-1)).sum()); cnt += a.shape[0]
    res = {"merge_seconds": merge_s, "max_abs_logit_diff": mx, "mean_abs_logit_diff": mean / len(prompts),
           "top1_agreement": agree / cnt, "positions": cnt, "merged_path": str(dst)}
    save_json(res, ad.parent / "merge_check.json")
    print(run, res, flush=True)
    del merged, unmerged
    torch.cuda.empty_cache()
print("Done.", flush=True)
