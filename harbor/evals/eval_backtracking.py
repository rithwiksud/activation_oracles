"""Backtracking eval (held out, verifier image only): the oracle reads the base model's activations at the end of a
prefix of its own reasoning (last 10 tokens) and must say whether the model is about to revise its line of reasoning.
Classes are balanced, so chance is 50%.

Usage: python eval_backtracking.py --oracle-lora <adapter dir> [--output results.json]
"""

import argparse
import json
from dataclasses import asdict

from ao_eval_common import VerbalizerInputInfo, load_base_model, run_oracle

QUESTION = "Answer with yes or no only. Is the model about to reconsider or revise its current line of reasoning?"

parser = argparse.ArgumentParser()
parser.add_argument("--oracle-lora", required=True)
parser.add_argument("--items", default="/opt/ao_heldout/backtracking.json")
parser.add_argument("--max-items", type=int, default=None, help="for smoke tests")
parser.add_argument("--output", default=None)
args = parser.parse_args()

items = json.load(open(args.items))[: args.max_items]
# The context is the rendered chat prompt plus partial thinking, tokenized verbatim (see ao_eval_common)
infos = [VerbalizerInputInfo(context_prompt=it["context"], verbalizer_prompt=QUESTION, ground_truth=it["label"]) for it in items]

model, tokenizer = load_base_model()
results = run_oracle(model, tokenizer, args.oracle_lora, None, infos, max_new_tokens=5)

predictions = []
for r in results:
    (response,) = r.segment_responses
    words = response.strip().lower().split()
    predictions.append(words[0].strip(".,!") if words else "")
correct = [p == it["label"] for p, it in zip(predictions, items, strict=True)]
summary = {
    "accuracy": sum(correct) / len(correct),
    "predicted_yes_rate": sum(p == "yes" for p in predictions) / len(predictions),
    "unparseable_rate": sum(p not in ("yes", "no") for p in predictions) / len(predictions),
    "n": len(correct),
}
print(json.dumps(summary, indent=2))

if args.output is not None:
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": [asdict(r) for r in results]}, f, indent=2)
