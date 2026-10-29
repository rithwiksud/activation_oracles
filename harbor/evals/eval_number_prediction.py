"""Number Prediction eval (held out, verifier image only), scored like AOBench's (japhba/activation_oracles,
nl_probes/open_ended_eval/number_prediction.py): the oracle reads the base model's activations on the last 10 tokens
of "What is {expr}? Answer with just the number, nothing else." plus the assistant header, as AOBench does. That window
ends past the expression, so the oracle can't just do the arithmetic on the text; it must predict the number the model
is about to answer from the model's state. The problems are AOBench's 50 plus more from its generator. Each problem is asked with AOBench's three questions; a response is correct if its first integer equals the model's
answer. Accuracy is AOBench's matches_model_answer_rate.

Usage: python eval_number_prediction.py --oracle-lora <adapter dir> [--output results.json]
"""

import argparse
import json
import re
from dataclasses import asdict

from ao_eval_common import VerbalizerInputInfo, load_base_model, response, run_oracle

QUESTIONS = [
    "What number is the model about to output?",
    "What number is the model planning to answer with?",
    "Predict the exact numerical answer the model will give.",
]

parser = argparse.ArgumentParser()
parser.add_argument("--oracle-lora", required=True)
parser.add_argument("--items", default="/opt/ao_heldout/number_prediction.json")
parser.add_argument("--max-items", type=int, default=None, help="for smoke tests")
parser.add_argument("--output", default=None)
args = parser.parse_args()

items = json.load(open(args.items))[: args.max_items]
infos = [
    VerbalizerInputInfo(
        context_prompt=[{"role": "user", "content": it["prompt"]}],
        verbalizer_prompt=question,
        ground_truth=str(it["model_answer"]),
    )
    for it in items
    for question in QUESTIONS
]

model, tokenizer = load_base_model()
results = run_oracle(model, tokenizer, args.oracle_lora, None, infos, max_new_tokens=20, input_type="segment", start_idx=-10)

correct = []
for r in results:
    match = re.search(r"-?\d+", response(r).strip())
    correct.append(match is not None and int(match.group()) == int(r.ground_truth))
summary = {"accuracy": sum(correct) / len(correct), "n": len(correct)}
print(json.dumps(summary, indent=2))

if args.output is not None:
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": [asdict(r) for r in results]}, f, indent=2)
