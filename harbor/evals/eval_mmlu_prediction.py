"""MMLU Prediction eval (held out, verifier image only), scored like AOBench's (japhba/activation_oracles,
nl_probes/open_ended_eval/mmlu_prediction.py and run_all.py), except that the oracle reads the base model's activations on
the full sequence of an MMLU question (AOBench reads the last 10 tokens). It is asked either before the model answers
("Will the model answer this correctly?") or after its answer letter ("Is the model's answer likely to be correct?").
Each response is scored by the yes-minus-no logit margin of its first token. The score is the ROC AUC against whether
the model was actually correct, averaged over the two questions (chance 0.5).

Usage: python eval_mmlu_prediction.py --oracle-lora <adapter dir> [--output results.json]
"""

import argparse
import json

from ao_eval_common import VerbalizerInputInfo, load_base_model, roc_auc, run_oracle_yes_no

PREFIX = "Answer with 'Yes' or 'No' only. "
QUESTIONS = {
    "pre_answer": f"{PREFIX}Will the model answer this correctly?",
    "post_answer": f"{PREFIX}Is the model's answer likely to be correct?",
}
ANSWER_LETTERS = ["A", "B", "C", "D"]

parser = argparse.ArgumentParser()
parser.add_argument("--oracle-lora", required=True)
parser.add_argument("--items", default="/opt/ao_heldout/mmlu_prediction.json")
parser.add_argument("--max-items", type=int, default=None, help="for smoke tests")
parser.add_argument("--output", default=None)
args = parser.parse_args()

items = json.load(open(args.items))[: args.max_items]
model, tokenizer = load_base_model()


def question_text(it: dict) -> str:
    lines = [it["question"], ""] + [f"{ANSWER_LETTERS[i]}. {c}" for i, c in enumerate(it["choices"])]
    return "\n".join(lines + ["", "Answer with just the letter (A, B, C, or D), nothing else."])


def context(it: dict, mode: str) -> list[dict] | str:
    """Pre-answer: the user turn plus the assistant header. Post-answer: the finished turn with the model's letter,
    rendered without a generation prompt (a pre-rendered string is tokenized verbatim, see ao_eval_common)."""
    user = {"role": "user", "content": question_text(it)}
    if mode == "pre_answer":
        return [user]
    return tokenizer.apply_chat_template(
        [user, {"role": "assistant", "content": it["model_answer_letter"]}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


summary = {"by_question": {}}
all_margins = {}
for mode, question in QUESTIONS.items():
    infos = [
        VerbalizerInputInfo(context_prompt=context(it, mode), verbalizer_prompt=question,
                            ground_truth="yes" if it["model_correct"] else "no")
        for it in items
    ]  # fmt: skip
    margins = run_oracle_yes_no(model, tokenizer, args.oracle_lora, None, infos, input_type="full_seq")
    labels = [it["model_correct"] for it in items]
    summary["by_question"][mode] = {
        "auc": roc_auc(margins, labels),
        "accuracy_at_zero": sum((m >= 0) == y for m, y in zip(margins, labels, strict=True)) / len(items),
    }
    all_margins[mode] = margins
# The reported score (read as "accuracy" like every other eval) is the mean AUC
summary["accuracy"] = sum(q["auc"] for q in summary["by_question"].values()) / len(QUESTIONS)
summary["n"] = len(items)
print(json.dumps(summary, indent=2))

if args.output is not None:
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "margins": all_margins}, f, indent=2)
