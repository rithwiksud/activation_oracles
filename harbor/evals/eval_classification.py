"""OOD classification eval, the AO paper's protocol (experiments/classification_eval.py at commit 0606a87 and
experiments/plotting/plot_classification_eval.py): the paper's 7 held-out yes/no classification datasets (App. C.3),
activations of the full sequence (the paper reads a single token) at layers 25/50/75%, greedy answers of up to 10
tokens. A response
is correct if it contains the target ("yes" / "no"), case-insensitive, and accuracy is pooled over all items of all
datasets, as the paper's OOD average is. The test sets are built by harbor/scripts/build_classification_test_sets.py.

Usage: python eval_classification.py --oracle-lora <adapter dir> --data-dir <dir with the cached .pt test sets>
"""

import argparse
import glob
import json
import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from peft import PeftModel

from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint
from nl_probes.utils.eval import run_evaluation

MODEL_NAME = "Qwen/Qwen3.5-0.8B"

parser = argparse.ArgumentParser()
parser.add_argument("--oracle-lora", required=True)
parser.add_argument("--data-dir", required=True)
parser.add_argument(
    "--datasets",
    nargs="+",
    default=[
        "ag_news",
        "language_identification",
        "singular_plural",
        "engels_headline_istrump",
        "engels_headline_isobama",
        "engels_headline_ischina",
        "engels_hist_fig_ismale",
    ],
)
parser.add_argument("--max-items", type=int, default=None, help="per dataset; for smoke tests")
parser.add_argument("--output", default=None)
args = parser.parse_args()

tokenizer = load_tokenizer(MODEL_NAME)
model = PeftModel.from_pretrained(load_model(MODEL_NAME, torch.bfloat16), args.oracle_lora).eval()
submodule = get_hf_submodule(model, 1, use_lora=True)
# Stop at the end of the assistant turn (Qwen3.5's model config only stops at <|endoftext|>)
generation_kwargs = {
    "do_sample": False,
    "max_new_tokens": 10,
    "eos_token_id": sorted({model.generation_config.eos_token_id, tokenizer.eos_token_id}),
}

by_dataset = {}
for ds in args.datasets:
    files = sorted(glob.glob(f"{args.data_dir}/classification_{ds}_*test*.pt"))
    assert files, f"no cached test set for {ds} in {args.data_dir}"
    data = [TrainingDataPoint(**d) for f in files for d in torch.load(f, weights_only=False)["data"]][: args.max_items]
    responses = run_evaluation(
        eval_data=data, model=model, tokenizer=tokenizer, submodule=submodule, device=torch.device("cuda"),
        dtype=torch.bfloat16, global_step=-1, lora_path=None, eval_batch_size=256, steering_coefficient=1.0,
        generation_kwargs=generation_kwargs,
    )
    by_dataset[ds] = [
        dp.target_output.lower().strip() in r.api_response.lower().strip()
        for r, dp in zip(responses, data, strict=True)
    ]

pooled = [c for cs in by_dataset.values() for c in cs]
summary = {
    "accuracy": sum(pooled) / len(pooled),
    "by_dataset": {ds: sum(cs) / len(cs) for ds, cs in by_dataset.items()},
    "n": len(pooled),
}
print(json.dumps(summary, indent=2))
if args.output is not None:
    json.dump({"summary": summary}, open(args.output, "w"), indent=2)
