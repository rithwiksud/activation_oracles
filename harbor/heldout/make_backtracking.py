"""Held-out Backtracking data (AOBench-style): Qwen3.5-0.8B solves GSM8K problems in thinking mode. Each item is the
user question plus a prefix of the model's own reasoning, cut at a sentence boundary; the label says whether the next
sentence starts a revision ("Wait", "Actually", "Let me re-check", ...). The oracle reads the last 10 tokens of the
partial reasoning and must predict whether the model is about to revise. Classes are balanced 50/50.

Usage: python make_backtracking.py <out.json> <gsm8k split> <n_problems> <max_items> <seed>
"""

import json
import random
import re
import sys

import torch
from datasets import load_dataset

from nl_probes.utils.common import load_model, load_tokenizer

out_path, split, n_problems, max_items, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
REVISION = re.compile(
    r"^(wait|hmm|actually|hold on|but wait|oops|no[,.]|that'?s not right|that can'?t be right|i made a mistake|"
    r"let me (re-?check|recheck|reconsider|recalculate|re-?examine|redo|re-?evaluate|double-check))",
    re.IGNORECASE,
)

rng = random.Random(seed)
problems = load_dataset("openai/gsm8k", "main", split=split)["question"]
problems = rng.sample(list(problems), n_problems)

tokenizer = load_tokenizer(MODEL_NAME)
model = load_model(MODEL_NAME, torch.bfloat16).eval()
prompts = [
    tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    for q in problems
]

traces = []
for i in range(0, len(prompts), 128):
    batch = prompts[i : i + 128]
    enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=1024, do_sample=True, temperature=0.6, top_p=0.95, eos_token_id=tokenizer.eos_token_id)
    traces += tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=False)
    print(f"generated {len(traces)}/{len(prompts)}", flush=True)

positives, negatives = [], []
for prompt, trace in zip(prompts, traces, strict=True):
    thinking = trace.split("</think>")[0]
    # Sentence starts: after ". ", "? ", "! " or a newline
    starts = [0] + [m.end() for m in re.finditer(r"(?<=[.?!])\s+|\n+", thinking)]
    starts = sorted({s for s in starts if 40 <= s < len(thinking) - 5})
    sentences = [thinking[s:].lstrip() for s in starts]
    is_rev = [bool(REVISION.match(s)) for s in sentences]
    pos = [j for j in range(len(starts)) if is_rev[j]]
    neg = [j for j in range(len(starts) - 2) if not any(is_rev[j : j + 2])]
    if pos:
        j = rng.choice(pos)
        positives.append({"context": prompt + thinking[: starts[j]].rstrip(), "label": "yes"})
    if neg:
        j = rng.choice(neg)
        negatives.append({"context": prompt + thinking[: starts[j]].rstrip(), "label": "no"})

k = min(len(positives), len(negatives), max_items // 2)
items = rng.sample(positives, k) + rng.sample(negatives, k)
rng.shuffle(items)
json.dump(items, open(out_path, "w"), indent=1)
print(f"traces {len(traces)} | with a revision {len(positives)} | negatives {len(negatives)} | wrote {len(items)} balanced items to {out_path}")
