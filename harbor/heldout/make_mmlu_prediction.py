"""Held-out MMLU Prediction data, built like AOBench's (japhba/activation_oracles,
data_pipelines/mmlu_prediction/generate_dataset.py): Qwen3.5-0.8B answers every MMLU test question greedily with thinking
disabled, then questions it got right and questions it got wrong are sampled in equal numbers (same seed and shuffles
as AOBench). AOBench samples 50 of each; we take 500 of each, since 100 items put the AUC's standard error near 0.06. The oracle
later reads the activations of the question (before the answer) or of the question plus the model's answer letter
(after it), and must say whether the model is / was correct.

Usage: python make_mmlu_prediction.py <out.json>
"""

import json
import random
import re
import sys

import torch
from datasets import load_dataset

from nl_probes.utils.common import load_model, load_tokenizer

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
TARGET_PER_CLASS = 500
ANSWER_LETTERS = ["A", "B", "C", "D"]
BATCH_SIZE = 64

random.seed(42)


def format_mmlu_question(question: str, choices: list[str]) -> str:
    lines = [question, ""]
    for i, choice in enumerate(choices):
        lines.append(f"{ANSWER_LETTERS[i]}. {choice}")
    lines.append("")
    lines.append("Answer with just the letter (A, B, C, or D), nothing else.")
    return "\n".join(lines)


def extract_answer_letter(response: str) -> str | None:
    text = response.strip()
    if text and text[0] in "ABCD":
        return text[0]
    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1)
    return None


out_path = sys.argv[1]
ds = load_dataset("cais/mmlu", "all", split="test")
tokenizer = load_tokenizer(MODEL_NAME)
model = load_model(MODEL_NAME, torch.bfloat16).eval()

items = [
    {
        "index": i,
        "question": row["question"],
        "subject": row["subject"],
        "choices": row["choices"],
        "correct_answer_idx": row["answer"],
        "correct_answer_letter": ANSWER_LETTERS[row["answer"]],
    }
    for i, row in enumerate(ds)
]
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": format_mmlu_question(it["question"], it["choices"])}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    for it in items
]
# Generate in length-sorted batches (greedy decoding, so the order doesn't change the answers)
order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
responses: list[str | None] = [None] * len(prompts)
for start in range(0, len(order), BATCH_SIZE):
    idx = order[start : start + BATCH_SIZE]
    enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=10, do_sample=False, eos_token_id=tokenizer.eos_token_id)
    for i, text in zip(idx, tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)):
        responses[i] = text
    if start % (BATCH_SIZE * 40) == 0:
        print(f"answered {start + len(idx)}/{len(order)}", flush=True)

correct_items, incorrect_items = [], []
for it, raw in zip(items, responses, strict=True):
    letter = extract_answer_letter(raw)
    it.update(model_raw_response=raw, model_answer_letter=letter, model_correct=letter == it["correct_answer_letter"])
    if letter is None:
        continue
    (correct_items if it["model_correct"] else incorrect_items).append(it)

n_parseable = len(correct_items) + len(incorrect_items)
n_per_class = min(TARGET_PER_CLASS, len(correct_items), len(incorrect_items))
random.shuffle(correct_items)
random.shuffle(incorrect_items)
fields = ["question", "subject", "choices", "correct_answer_idx", "correct_answer_letter", "model_raw_response",
          "model_answer_letter", "model_correct"]  # fmt: skip
entries = [{"id": f"mmlu_{it['index']}", **{f: it[f] for f in fields}}
           for it in correct_items[:n_per_class] + incorrect_items[:n_per_class]]  # fmt: skip
random.shuffle(entries)
json.dump(entries, open(out_path, "w"), indent=1)
print(
    f"MMLU: {len(items)} questions, {n_parseable} parseable, model accuracy {len(correct_items) / n_parseable:.3f} | "
    f"wrote {len(entries)} balanced items to {out_path}"
)
