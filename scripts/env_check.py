"""Step 0 for PA4: record the environment and check the three models load and behave.

Run:  python scripts/env_check.py
Writes results/env/env_check.json and results/env/pip_freeze.txt
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import transformers
import huggingface_hub
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.utils import RESULTS_DIR, SEED, gpu_peak_gb, save_json, set_seed

BASE = "Qwen/Qwen2.5-1.5B"              # Task 1
TARGET = "Qwen/Qwen2.5-1.5B-Instruct"   # Task 2C + Task 3 target
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"    # Task 3 draft
out = {}


def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


# 1. Versions and hardware -----------------------------------------------------
section("1. Environment")
out["python"] = sys.version.split()[0]
out["torch"] = torch.__version__
out["cuda"] = torch.version.cuda
out["transformers"] = transformers.__version__
out["huggingface_hub"] = huggingface_hub.__version__
for pkg in ["peft", "accelerate", "datasets", "scipy"]:
    try:
        out[pkg] = __import__(pkg).__version__
    except Exception as e:  # noqa: BLE001
        out[pkg] = f"MISSING ({e.__class__.__name__})"
out["gpus"] = [{"name": torch.cuda.get_device_name(i),
                "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
                "mem_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 1)}
               for i in range(torch.cuda.device_count())]
for k, v in out.items():
    print(f"{k:16s} {v}")
assert torch.cuda.is_available(), "No GPU. Set Accelerator = GPU T4 x2."
assert transformers.__version__ == "4.57.1", "Wrong transformers version: pip install -r requirements.txt"

env_dir = RESULTS_DIR / "env"
env_dir.mkdir(parents=True, exist_ok=True)
(env_dir / "pip_freeze.txt").write_text(
    subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)

# 2. Exact checkpoint revisions (the guide asks us to record these) ----------
section("2. Model revisions")
out["revisions"] = {}
for m in [BASE, TARGET, DRAFT]:
    try:
        out["revisions"][m] = model_info(m).sha
    except Exception as e:  # noqa: BLE001
        out["revisions"][m] = f"ERROR {e.__class__.__name__}"
    print(f"{m:30s} {out['revisions'][m]}")

# 3. SDPA backends on this GPU (Task 2A) --------------------------------------
section("3. SDPA backends (B=2, H=4, L=1024, d=64)")
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402
import torch.nn.functional as F  # noqa: E402

q = torch.randn(2, 4, 1024, 64, device="cuda")
back = {}
for dtype in [torch.float16, torch.float32]:
    for name, b in [("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                    ("math", SDPBackend.MATH)]:
        try:
            with sdpa_kernel(b):
                x = q.to(dtype)
                F.scaled_dot_product_attention(x, x, x, is_causal=True)
            back[f"{name}_{str(dtype)[6:]}"] = "ok"
        except Exception as e:  # noqa: BLE001
            back[f"{name}_{str(dtype)[6:]}"] = f"unsupported ({str(e).splitlines()[0][:60]})"
for k, v in back.items():
    print(f"{k:20s} {v}")
out["sdpa_backends"] = back

# 4. Tokenizers: target and draft must agree (Task 3) --------------------------
section("4. Tokenizer / chat-template agreement (target vs draft)")
tt, td = AutoTokenizer.from_pretrained(TARGET), AutoTokenizer.from_pretrained(DRAFT)
msgs = [{"role": "user", "content": "What is 17 * 3? Answer briefly."}]
pt = tt.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
pd_ = td.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
out["chat_prompt_target"] = pt
out["chat_templates_equal"] = pt == pd_
out["token_ids_equal"] = tt(pt)["input_ids"] == td(pd_)["input_ids"]
out["vocab_len_target_draft"] = [len(tt), len(td)]
out["eos_target"] = [tt.eos_token, tt.eos_token_id]
print(repr(pt))
print("templates equal:", out["chat_templates_equal"], "| ids equal:", out["token_ids_equal"],
      "| len(tokenizer):", out["vocab_len_target_draft"], "| eos:", out["eos_target"])

# 5. Base model: structure, tied weights, eligible projections (Task 1) --------
section("5. Qwen2.5-1.5B base: eligible weights and speed")
set_seed(SEED)
tb = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).cuda().eval()
cfg = model.config
out["base_config"] = {k: getattr(cfg, k) for k in
                      ["num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads",
                       "num_key_value_heads", "vocab_size", "tie_word_embeddings"]}
out["attn_implementation"] = getattr(cfg, "_attn_implementation", None)
names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
elig = {n: 0 for n in names}
for n, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear) and n.split(".")[-1] in names:
        elig[n.split(".")[-1]] += mod.weight.numel()
seen, total = set(), 0
for p in model.parameters():
    if id(p) not in seen:
        seen.add(id(p))
        total += p.numel()
out["eligible_weights_by_type"] = elig
out["eligible_weights_total"] = sum(elig.values())
out["all_params_tied_once"] = total
out["lm_head_tied_to_embed"] = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
print(out["base_config"])
print("attention backend:", out["attn_implementation"])
print("eligible weights:", f"{out['eligible_weights_total']:,}", "of", f"{total:,}",
      "| lm_head tied:", out["lm_head_tied_to_embed"])

# one 512-token window: NLL and timing
x = torch.randint(100, 20000, (1, 512), device="cuda")
with torch.no_grad():
    for _ in range(2):
        model(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        model(x)
    torch.cuda.synchronize()
out["base_prefill512_ms"] = round((time.time() - t0) / 5 * 1000, 1)
ids = tb("The capital of France is", return_tensors="pt").to("cuda")
with torch.no_grad():
    g = model.generate(**ids, max_new_tokens=20, do_sample=False)
out["base_sample"] = tb.decode(g[0, ids["input_ids"].shape[1]:])
out["base_peak_gb"] = gpu_peak_gb(reset=True)
print("prefill 512 tokens:", out["base_prefill512_ms"], "ms | peak", out["base_peak_gb"], "GB")
print("sample:", repr(out["base_sample"]))

save_json(out, env_dir / "env_check.json")
section("Saved results/env/env_check.json")
