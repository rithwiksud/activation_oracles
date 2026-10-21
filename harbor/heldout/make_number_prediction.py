"""Held-out Number Prediction data (AOBench-style): the oracle reads the base model's activations on an arithmetic
question and must name the numbers in it. The read window (last 10 tokens) is the "?" plus the chat template's
assistant header, so the digits themselves are not in it: the numbers have to come from the residual stream.

Usage: python make_number_prediction.py <out.json> <n> <seed>
"""

import json
import random
import sys

out_path, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rng = random.Random(seed)
TEMPLATES = ["What is {a} {op} {b}?", "Compute {a} {op} {b}.", "Calculate {a} {op} {b}.", "What does {a} {op} {b} equal?"]

items = []
for _ in range(n):
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    op = rng.choice(["+", "-"])
    items.append({"a": a, "b": b, "op": op, "prompt": rng.choice(TEMPLATES).format(a=a, op=op, b=b)})
json.dump(items, open(out_path, "w"), indent=1)
print(f"wrote {n} items to {out_path}")
