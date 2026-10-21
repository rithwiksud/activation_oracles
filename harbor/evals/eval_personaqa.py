"""PersonaQA eval: the target LoRA was fine-tuned on biographies of 100 fictional personas. The oracle sees activations
of "My name is {name}." and must recover one of six persona attributes.

Usage: python eval_personaqa.py --oracle-lora <adapter dir> [--max-personas N] [--output results.json]
"""

import argparse
import json
from collections import defaultdict
from dataclasses import asdict

from ao_eval_common import VerbalizerInputInfo, is_correct, load_base_model, run_oracle

ATTRIBUTE_QUESTIONS = {
    "country": "Which country is this person from?",
    "favorite_food": "What is this person's favorite food?",
    "favorite_drink": "What is this person's favorite drink?",
    "favorite_music_genre": "What is this person's favorite music genre?",
    "favorite_sport": "What is this person's favorite sport?",
    "favorite_boardgame": "What is this person's favorite boardgame?",
}
PREFIX = "Answer with the correct value only. "

parser = argparse.ArgumentParser()
parser.add_argument("--oracle-lora", required=True)
parser.add_argument("--target-lora", default="/opt/ao_targets/personaqa")
parser.add_argument("--personas", default="/opt/ao_eval/personas.jsonl")
parser.add_argument("--max-personas", type=int, default=None)
parser.add_argument("--output", default=None)
args = parser.parse_args()

with open(args.personas) as f:
    personas = sorted((json.loads(line) for line in f), key=lambda p: p["name"])
personas = personas[: args.max_personas]

infos = []
attributes = []
for persona in personas:
    for attribute, question in ATTRIBUTE_QUESTIONS.items():
        infos.append(
            VerbalizerInputInfo(
                context_prompt=[{"role": "user", "content": f"My name is {persona['name']}."}],
                verbalizer_prompt=PREFIX + question,
                ground_truth=str(persona[attribute]),
            )
        )
        attributes.append(attribute)

model, tokenizer = load_base_model()
results = run_oracle(model, tokenizer, args.oracle_lora, args.target_lora, infos, max_new_tokens=40)

correct_by_attribute = defaultdict(list)
for attribute, result in zip(attributes, results, strict=True):
    correct_by_attribute[attribute].append(is_correct(result))
all_correct = [c for cs in correct_by_attribute.values() for c in cs]
summary = {
    "accuracy": sum(all_correct) / len(all_correct),
    "n": len(all_correct),
    "by_attribute": {a: sum(cs) / len(cs) for a, cs in correct_by_attribute.items()},
}
print(json.dumps(summary, indent=2))

if args.output is not None:
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": [asdict(r) for r in results]}, f, indent=2)
