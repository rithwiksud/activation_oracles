"""Number Prediction eval (held out, verifier image only): the oracle reads the base model's activations on an arithmetic
question (last 10 tokens, which exclude the digits) and must name the two operands. Score per item = fraction of the
two operands that appear in the answer as whole numbers.

Usage: python eval_number_prediction.py --oracle-lora <adapter dir> [--output results.json]
"""

import argparse
import json
import re
from dataclasses import asdict

from ao_eval_common import VerbalizerInputInfo, load_base_model, run_oracle

QUESTION = "Answer with the numbers only. What numbers appear in the arithmetic problem?"

parser = argparse.ArgumentParser()
parser.add_argument("--oracle-lora", required=True)
parser.add_argument("--items", default="/opt/ao_heldout/number_prediction.json")
parser.add_argument("--output", default=None)
args = parser.parse_args()

items = json.load(open(args.items))
infos = [
    VerbalizerInputInfo(
        context_prompt=[{"role": "user", "content": it["prompt"]}],
        verbalizer_prompt=QUESTION,
        ground_truth=f"{it['a']} {it['b']}",
    )
    for it in items
]

model, tokenizer = load_base_model()
results = run_oracle(model, tokenizer, args.oracle_lora, None, infos, max_new_tokens=20)

scores = []
for it, r in zip(items, results, strict=True):
    (response,) = r.segment_responses
    found = [bool(re.search(rf"(?<!\d){n}(?!\d)", response)) for n in (it["a"], it["b"])]
    scores.append(sum(found) / 2)
summary = {
    "accuracy": sum(scores) / len(scores),
    "both_correct": sum(s == 1 for s in scores) / len(scores),
    "n": len(scores),
}
print(json.dumps(summary, indent=2))

if args.output is not None:
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": [asdict(r) for r in results]}, f, indent=2)
