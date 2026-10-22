"""Held-out Number Prediction data, built exactly like AOBench's (japhba/activation_oracles,
data_pipelines/number_prediction/generate_dataset.py): the same 50 arithmetic problems (same generator and seed, checked
against AOBench's Qwen3-8B dataset when its path is given), answered by Qwen3.5-0.8B greedily with thinking disabled. The
oracle later reads the activations of the question and must predict the model's own answer.

Usage: python make_number_prediction.py <out.json> [<AOBench Qwen3-8B number_prediction_eval_dataset.json>]
"""

import json
import random
import re
import sys

import torch

from nl_probes.utils.common import load_model, load_tokenizer

MODEL_NAME = "Qwen/Qwen3.5-0.8B"


def user_prompt(expression: str) -> str:
    return f"What is {expression}? Answer with just the number, nothing else."


# Copied verbatim from AOBench (the random call sequence defines the problems)
random.seed(42)


def generate_problems():
    problems = []

    # Simple 2-operand (answers likely 1-2 tokens)
    for i in range(10):
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        op = random.choice(["+", "-", "*"])
        expr = f"{a} {op} {b}"
        problems.append({"id": f"simple_{i}", "expression": expr, "true_answer": eval(expr), "category": "simple_2op"})

    # Medium complexity (3-4 operands, mixed ops)
    for i in range(15):
        n_ops = random.randint(2, 3)
        nums = [random.randint(-99, 99) for _ in range(n_ops + 1)]
        ops = [random.choice(["+", "-", "*"]) for _ in range(n_ops)]
        expr = str(nums[0])
        for j in range(n_ops):
            if random.random() < 0.3:
                expr = f"({expr} {ops[j]} {nums[j+1]})"
            else:
                expr = f"{expr} {ops[j]} {nums[j+1]}"
        problems.append({"id": f"medium_{i}", "expression": expr, "true_answer": eval(expr), "category": "medium_3_4op"})

    # Large 2-operand (answers likely multi-token)
    for i in range(10):
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        op = random.choice(["+", "-", "*"])
        expr = f"{a} {op} {b}"
        problems.append({"id": f"large_{i}", "expression": expr, "true_answer": eval(expr), "category": "large_numbers"})

    # Division and modulo
    for i in range(8):
        a = random.randint(50, 500)
        b = random.randint(2, 20)
        op = random.choice(["//", "%"])
        expr = f"{a} {op} {b}"
        problems.append({"id": f"divmod_{i}", "expression": expr, "true_answer": eval(expr), "category": "divmod"})

    # Nested expressions
    for i in range(7):
        nums = [random.randint(-99, 99) for _ in range(5)]
        templates = [
            f"({nums[0]} * {nums[1]}) + ({nums[2]} - {nums[3]}) * {nums[4]}",
            f"({nums[0]} + {nums[1]}) * ({nums[2]} - {nums[3]})",
            f"{nums[0]} * {nums[1]} + {nums[2]} * {nums[3]} - {nums[4]}",
            f"({nums[0]} + {nums[1]} + {nums[2]}) * ({nums[3]} - {nums[4]})",
            f"{nums[0]} * ({nums[1]} + {nums[2]}) - {nums[3]} * {nums[4]}",
        ]
        expr = random.choice(templates)
        problems.append({"id": f"nested_{i}", "expression": expr, "true_answer": eval(expr), "category": "nested"})

    return problems


out_path = sys.argv[1]
problems = generate_problems()
if len(sys.argv) > 2:
    reference = [e["expression"] for e in json.load(open(sys.argv[2]))["entries"]]
    assert [p["expression"] for p in problems] == reference, "problems differ from AOBench's"

tokenizer = load_tokenizer(MODEL_NAME)
model = load_model(MODEL_NAME, torch.bfloat16).eval()
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt(p["expression"])}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    for p in problems
]
enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=50, do_sample=False, eos_token_id=tokenizer.eos_token_id)
responses = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)

entries = []
for p, raw in zip(problems, responses, strict=True):
    match = re.search(r"-?\d+", raw.strip())
    model_answer = int(match.group()) if match else None
    entries.append(
        {
            **p,
            "prompt": user_prompt(p["expression"]),
            "model_raw_response": raw,
            "model_answer": model_answer,
            "model_correct": model_answer == p["true_answer"],
            "is_single_token_answer": len(tokenizer.encode(str(p["true_answer"]), add_special_tokens=False)) == 1,
        }
    )
json.dump(entries, open(out_path, "w"), indent=1)
n_correct = sum(e["model_correct"] for e in entries)
n_missing = sum(e["model_answer"] is None for e in entries)
print(f"wrote {len(entries)} items to {out_path} | model correct {n_correct} | no number in response {n_missing}")
