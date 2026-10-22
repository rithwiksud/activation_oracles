"""Training/eval logging helpers that make the wandb curves and examples readable.

Raw per-step loss swings a lot because batches are length-grouped, and length correlates with task (a one-token yes/no
answer has much lower loss than a 50-token passage). So we also log token-weighted losses per task over a full
megabatch cycle, and tables of example inputs/outputs.
"""

import re

import torch
import wandb
from transformers import AutoTokenizer

from nl_probes.utils.dataset_utils import TrainingDataPoint


def task_group(datapoint_type: str) -> str:
    for prefix in ("classification", "latentqa"):
        if datapoint_type.startswith(prefix):
            return prefix
    return datapoint_type


@torch.no_grad()
def per_example_token_losses(logits: torch.Tensor, labels: torch.Tensor, chunk: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed next-token loss and target-token count per example (the model's loss is their ratio, pooled)."""
    shift_logits, shift_labels = logits[:, :-1], labels[:, 1:]
    mask = shift_labels != -100
    rows, cols = mask.nonzero(as_tuple=True)
    token_losses = torch.cat([
        torch.nn.functional.cross_entropy(shift_logits[rows[i : i + chunk], cols[i : i + chunk]].float(),
                                          shift_labels[rows[i : i + chunk], cols[i : i + chunk]], reduction="none")
        for i in range(0, len(rows), chunk)
    ])
    sums = torch.zeros(labels.shape[0], device=labels.device).index_add_(0, rows, token_losses)
    return sums, mask.sum(1)


def format_example(dp: TrainingDataPoint, tokenizer: AutoTokenizer) -> dict[str, str]:
    """Target-model context with the positions the oracle reads marked ⟦like this⟧, the oracle's question, and the
    target answer."""
    marked = set(dp.context_positions)
    pieces = []
    for i, t in enumerate(dp.context_input_ids):
        if i in marked and i - 1 not in marked:
            pieces.append("⟦")
        pieces.append(tokenizer.decode([t]))
        if i in marked and i + 1 not in marked:
            pieces.append("⟧")
    context = "".join(pieces)
    prompt_ids = [t for t, l in zip(dp.input_ids, dp.labels) if l == -100]
    question = tokenizer.decode(prompt_ids, skip_special_tokens=True)
    question = re.sub(r"(?: \?)+", f" ⟨{len(dp.positions)} activations⟩", question)
    # Drop chat-template role names and the empty thinking block left after removing special tokens
    question = re.sub(r"^\s*user\s*|\s*assistant\s*(?:<think>\s*</think>)?\s*$", "", question)
    target = tokenizer.decode([l for l in dp.labels if l != -100], skip_special_tokens=True).strip()
    return {"context (read positions ⟦marked⟧)": context, "oracle question": question, "target": target}


def examples_table(rows: list[dict]) -> wandb.Table:
    columns = list(rows[0])
    return wandb.Table(columns=columns, data=[[r[c] for c in columns] for r in rows])
