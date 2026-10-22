"""Shared scoring for the Harbor activation oracle evals.

The oracle reads activations from a target LoRA (a fine-tune that holds hidden information), or from the base model, and
answers an open-ended question. A response counts as correct if it contains the ground truth, case-insensitive. Every
eval reads the full sequence (every context token), the AO paper's default input. Prompts and scoring follow each
task's source paper.
"""

import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from peft import LoraConfig

import nl_probes.base_experiment as base_experiment
import nl_probes.utils.eval as eval_utils
from nl_probes.base_experiment import VerbalizerInputInfo, VerbalizerResults
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.dataset_utils import FeatureResult
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook

MODEL_NAME = "Qwen/Qwen3.5-0.8B"


def _encode_contexts(tokenizer, message_dicts, add_generation_prompt: bool, enable_thinking: bool, device):
    """base_experiment.encode_messages, extended so a context can also be a pre-rendered string (used verbatim). This
    lets evals read activations from inside a partial assistant turn."""
    texts = [
        source
        if isinstance(source, str)
        else tokenizer.apply_chat_template(
            source, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking
        )
        for source in message_dicts
    ]
    return tokenizer(texts, return_tensors="pt", add_special_tokens=False, padding=True).to(device)


base_experiment.encode_messages = _encode_contexts


def load_base_model():
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, torch.bfloat16)
    model.eval()
    # Dummy adapter so peft_config exists and the PeftModel adapter API works. PEFT has no default target modules for
    # Qwen3.5, and every decoder layer has an MLP down_proj.
    model.add_adapter(LoraConfig(target_modules=["down_proj"]), adapter_name="default")
    return model, tokenizer


@torch.no_grad()
def run_oracle(
    model,
    tokenizer,
    oracle_lora: str,
    target_lora: str | None,
    infos: list[VerbalizerInputInfo],
    max_new_tokens: int,
    input_type: str,
    start_idx: int | None = None,
) -> list[VerbalizerResults]:
    """target_lora=None reads the base model's own activations (adapters disabled). input_type is "tokens" (the single
    token at start_idx), "segment" (start_idx to the end) or "full_seq"."""
    assert (start_idx is None) == (input_type == "full_seq"), (input_type, start_idx)
    positions = {}
    if input_type == "tokens":
        positions = {"token_start_idx": start_idx, "token_end_idx": start_idx + 1}
    elif input_type == "segment":
        positions = {"segment_start_idx": start_idx, "segment_end_idx": 0}
    config = base_experiment.VerbalizerEvalConfig(
        model_name=MODEL_NAME,
        activation_input_types=["orig"] if target_lora is None else ["lora"],
        verbalizer_input_types=[input_type],
        **positions,
        eval_batch_size=512,
        # Qwen3.5-0.8B ships no generation_config and its model config's eos is <|endoftext|>, but the oracle is trained
        # to end answers with <|im_end|> (the tokenizer's eos), so stop on that
        verbalizer_generation_kwargs={
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "eos_token_id": tokenizer.eos_token_id,
        },
        full_seq_repeats=1,
        segment_repeats=1,
    )
    oracle_name = base_experiment.load_lora_adapter(model, oracle_lora)
    target_name = None if target_lora is None else base_experiment.load_lora_adapter(model, target_lora)
    results = base_experiment.run_verbalizer(
        model=model,
        tokenizer=tokenizer,
        verbalizer_prompt_infos=infos,
        verbalizer_lora_path=oracle_name,
        target_lora_path=target_name,
        config=config,
        device=torch.device("cuda"),
    )
    if target_name is not None:
        model.delete_adapter(target_name)
    return results


def response(result: VerbalizerResults) -> str:
    """The oracle's one response, whichever input type produced it."""
    (r,) = [t for t in result.token_responses if t is not None] + result.segment_responses + result.full_sequence_responses
    return r


def is_correct(result: VerbalizerResults) -> bool:
    return result.ground_truth.lower() in response(result).lower()


# AOBench's single-token yes/no variants (nl_probes/open_ended_eval/eval_runner.py)
YES_NO_VARIANTS = {
    "yes": ["yes", " yes", "Yes", " Yes", "YES", " YES", "\nyes", "\nYes", "\nYES"],
    "no": ["no", " no", "No", " No", "NO", " NO", "\nno", "\nNo", "\nNO"],
}


def run_oracle_yes_no(
    model,
    tokenizer,
    oracle_lora: str,
    target_lora: str | None,
    infos: list[VerbalizerInputInfo],
    input_type: str,
    start_idx: int | None = None,
) -> list[float]:
    """AOBench's binary scoring: instead of generating, score the oracle's first answer token as logsumexp over the
    logits of single-token "yes" variants minus the same over "no" variants. Returns one margin per info."""
    groups = {}
    for label, variants in YES_NO_VARIANTS.items():
        ids = [tokenizer.encode(v, add_special_tokens=False) for v in variants]
        groups[label] = sorted({i[0] for i in ids if len(i) == 1})
        assert groups[label], label

    @torch.no_grad()
    def score_batch(eval_batch, model, submodule, tokenizer, device, dtype, steering_coefficient, generation_kwargs):
        hook_fn = get_hf_activation_steering_hook(
            vectors=eval_batch.steering_vectors,
            positions=eval_batch.positions,
            steering_coefficient=steering_coefficient,
            device=device,
            dtype=dtype,
        )
        with add_hook(submodule, hook_fn):
            out = model(input_ids=eval_batch.input_ids, attention_mask=eval_batch.attention_mask, logits_to_keep=1)
        logits = out.logits[:, -1].float()
        margins = torch.logsumexp(logits[:, groups["yes"]], -1) - torch.logsumexp(logits[:, groups["no"]], -1)
        return [
            FeatureResult(feature_idx=idx, api_response=str(m), prompt="")
            for idx, m in zip(eval_batch.feature_indices, margins.tolist(), strict=True)
        ]

    # run_evaluation looks up eval_features_batch at call time, so this swaps generation for scoring
    generate_batch = eval_utils.eval_features_batch
    eval_utils.eval_features_batch = score_batch
    try:
        results = run_oracle(model, tokenizer, oracle_lora, target_lora, infos, 1, input_type, start_idx)
    finally:
        eval_utils.eval_features_batch = generate_batch
    return [float(response(r)) for r in results]


def roc_auc(scores: list[float], labels: list[bool]) -> float:
    """Probability that a random positive outscores a random negative (ties count half)."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    assert pos and neg
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))
