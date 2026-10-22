import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import math
import random
from collections import defaultdict
import time
from datetime import timedelta

# All necessary imports are now included above
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, BitsAndBytesConfig
from transformers.optimization import get_linear_schedule_with_warmup
import torch.distributed as dist
import wandb

import nl_probes.dataset_classes.classification as classification
from nl_probes.utils.steering_hooks import (
    add_hook,
    get_hf_activation_steering_hook,
)
from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.dataset_classes.act_dataset_manager import ActDatasetLoader, DatasetLoaderConfig
from nl_probes.dataset_classes.classification import (
    ClassificationDatasetConfig,
    ClassificationDatasetLoader,
)
from nl_probes.dataset_classes.latentqa_dataset import LatentQADatasetConfig, LatentQADatasetLoader
from nl_probes.dataset_classes.past_lens_dataset import PastLensDatasetConfig, PastLensDatasetLoader
from nl_probes.dataset_classes.sae_training_data import (
    SAEActivatingSequencesDatasetConfig,
    SAEActivatingSequencesDatasetLoader,
    SAEExplanationDatasetConfig,
    SAEExplanationDatasetLoader,
    SAEYesNoDatasetConfig,
    SAEYesNoDatasetLoader,
)
from nl_probes.utils.activation_utils import get_hf_submodule, get_text_only_lora_targets
from nl_probes.utils.common import load_model, load_tokenizer, set_seed
from nl_probes.utils.dataset_utils import (
    BatchData,
    EvalStepResult,
    FeatureResult,
    TrainingDataPoint,
    construct_batch,
    materialize_missing_steering_vectors,
)
from nl_probes.utils.eval import parse_answer, run_evaluation, score_eval_responses
from nl_probes.utils.wandb_logging import examples_table, format_example, per_example_token_losses, task_group


def push_lora_to_hf(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    repo_id: str,
    private: bool,
    commit_message: str = "Upload LoRA adapter after training",
) -> None:
    """
    Push the trained LoRA adapter to Hugging Face Hub.

    Args:
        model: The trained model with LoRA adapters
        tokenizer: The tokenizer used with the model
        repo_id: HuggingFace repository ID (e.g., "username/repo-name")
        commit_message: Commit message for the upload
        private: Whether to make the repository private

    Returns:
        bool: True if successful, False otherwise
    """

    print(f"Pushing LoRA adapter to Hugging Face Hub: {repo_id}")

    # Get the original model name to copy config from
    original_model_name = model.config._name_or_path
    if hasattr(model, "base_model"):
        # For LoRA models, get the base model name
        original_model_name = model.base_model.config._name_or_path

    # Push the model (LoRA adapters)
    model.push_to_hub(
        repo_id=repo_id,
        commit_message=commit_message,
        private=private,
    )

    # Push the tokenizer as well
    tokenizer.push_to_hub(
        repo_id=repo_id,
        commit_message=f"Upload tokenizer - {commit_message}",
        private=private,
    )

    # Copy config.json from the original model
    try:
        import tempfile

        from huggingface_hub import hf_hub_download, upload_file

        print(f"Copying config.json from original model: {original_model_name}")

        # Download config.json from the original model
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".json", delete=False) as tmp_file:
            config_path = hf_hub_download(
                repo_id=original_model_name,
                filename="config.json",
                cache_dir=None,
                force_download=False,
            )

            # Copy the file content
            with open(config_path, "rb") as src:
                tmp_file.write(src.read())
            tmp_file.flush()

            # Upload to the LoRA repo
            upload_file(
                path_or_fileobj=tmp_file.name,
                path_in_repo="config.json",
                repo_id=repo_id,
                commit_message=f"Copy config.json from {original_model_name}",
            )

        # Clean up temp file
        os.unlink(tmp_file.name)
        print(f"Successfully copied config.json from {original_model_name}")

    except Exception as e:
        print(f"Warning: Failed to copy config.json from original model: {e}")
        print("LoRA adapter uploaded successfully, but without original model config")

    # Create and upload README with base model metadata
    try:
        print("Creating README with base model metadata...")

        readme_content = f"""---
base_model: {original_model_name}
library_name: peft
---

# LoRA Adapter for SAE Introspection

This is a LoRA (Low-Rank Adaptation) adapter trained for SAE (Sparse Autoencoder) introspection tasks.

## Base Model
- **Base Model**: `{original_model_name}`
- **Adapter Type**: LoRA
- **Task**: SAE Feature Introspection

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("{original_model_name}")
tokenizer = AutoTokenizer.from_pretrained("{original_model_name}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")
```

## Training Details
This adapter was trained using the lightweight SAE introspection training script to help the model understand and explain SAE features through activation steering.
"""

        # Create temporary README file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp_readme:
            tmp_readme.write(readme_content)
            tmp_readme.flush()

            # Upload README to the LoRA repo
            upload_file(
                path_or_fileobj=tmp_readme.name,
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message="Add README with base model metadata",
            )

        # Clean up temp file
        os.unlink(tmp_readme.name)
        print("Successfully uploaded README with base model metadata")

    except Exception as e:
        print(f"Warning: Failed to upload README: {e}")
        print("LoRA adapter uploaded successfully, but without README")

    print(f"Successfully pushed LoRA adapter to: https://huggingface.co/{repo_id}")


def train_features_batch(
    cfg: SelfInterpTrainingConfig,
    training_batch: BatchData,
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Forward pass on a single batch with activations injected. Returns the model output (loss and logits).
    """

    batch_steering_vectors = training_batch.steering_vectors
    batch_positions = training_batch.positions

    # 3. Create and apply the activation steering hook
    hook_fn = get_hf_activation_steering_hook(
        vectors=batch_steering_vectors,
        positions=batch_positions,
        steering_coefficient=cfg.steering_coefficient,
        device=device,
        dtype=dtype,
    )

    tokenized_input = {
        "input_ids": training_batch.input_ids,
        "attention_mask": training_batch.attention_mask,
    }

    with add_hook(submodule, hook_fn):
        outputs = model(**tokenized_input, labels=training_batch.labels)

    return outputs


def eval_generation_kwargs(cfg: SelfInterpTrainingConfig, model, tokenizer: PreTrainedTokenizer) -> dict:
    """Stop at the end of the assistant turn. Qwen3.5 ships no generation_config and its model config only stops at
    <|endoftext|>, so a trained oracle (whose targets end in <|im_end|>\\n) runs on into a new turn and fails the exact
    yes/no format check. Add the tokenizer's eos (<|im_end|>) to whatever the model already stops on."""
    model_eos = model.generation_config.eos_token_id
    model_eos = model_eos if isinstance(model_eos, list) else [model_eos]
    return {**cfg.generation_kwargs, "eos_token_id": sorted(set(model_eos) | {tokenizer.eos_token_id})}


def training_examples_table(
    cfg: SelfInterpTrainingConfig,
    points: list[TrainingDataPoint],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
):
    """The oracle's current answers on a fixed sample of training examples, to watch them change over training."""
    model.eval()
    responses = run_evaluation(
        eval_data=points, model=model, tokenizer=tokenizer, submodule=submodule, device=device, dtype=dtype,
        global_step=global_step, lora_path=None, eval_batch_size=cfg.eval_batch_size,
        steering_coefficient=cfg.steering_coefficient, generation_kwargs=eval_generation_kwargs(cfg, model, tokenizer),
    )
    model.train()
    rows = [
        {"step": global_step, "task": dp.datapoint_type, **format_example(dp, tokenizer), "oracle answer": r.api_response}
        for dp, r in zip(points, responses, strict=True)
    ]
    return examples_table(rows)


def eval_all_datasets(
    cfg: SelfInterpTrainingConfig,
    eval_datasets: dict[str, list[TrainingDataPoint]],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
) -> None:
    model.eval()
    generation_kwargs = eval_generation_kwargs(cfg, model, tokenizer)
    eval_results = {}
    example_rows = []
    for ds in eval_datasets:
        eval_responses = run_evaluation(
            eval_data=eval_datasets[ds],
            model=model,
            tokenizer=tokenizer,
            submodule=submodule,
            device=device,
            dtype=dtype,
            global_step=global_step,
            lora_path=None,
            eval_batch_size=cfg.eval_batch_size,
            steering_coefficient=cfg.steering_coefficient,
            generation_kwargs=generation_kwargs,
        )
        percent_format_correct, percent_ans_correct = score_eval_responses(eval_responses, eval_datasets[ds])
        eval_results[f"eval_format_correct/{ds}"] = percent_format_correct
        eval_results[f"eval_ans_correct/{ds}"] = percent_ans_correct
        print(f"Step {global_step} {ds} format correct: {percent_format_correct}, ans correct: {percent_ans_correct}")
        for dp, r in list(zip(eval_datasets[ds], eval_responses))[: cfg.eval_examples_per_dataset]:
            correct = parse_answer(r.api_response) == parse_answer(dp.target_output)
            example_rows.append(
                {"step": global_step, "dataset": ds, **format_example(dp, tokenizer), "oracle answer": r.api_response,
                 "correct": "✓" if correct else "✗"}
            )

    wandb.log({**eval_results, "eval_examples": examples_table(example_rows)}, step=global_step)
    wandb.summary.update(eval_results)
    model.train()

    # Have occasionally seen OOMs on first training step after eval, so clear cache here
    torch.cuda.empty_cache()
    gc.collect()


def oom_preflight_check(
    cfg: SelfInterpTrainingConfig,
    training_data: list[TrainingDataPoint],
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    longest_prompt = max(training_data, key=lambda x: len(x.input_ids))
    num_rows = cfg.train_batch_size
    if cfg.train_tokens_per_batch is not None:
        num_rows = min(num_rows, cfg.train_tokens_per_batch // len(longest_prompt.input_ids))
    long_prompts = [longest_prompt] * num_rows
    long_prompts = materialize_missing_steering_vectors(long_prompts, tokenizer, model)
    largest_possible_batch = construct_batch(long_prompts, tokenizer, device)

    dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)

    for _ in tqdm(range(3), desc="OOM preflight check"):
        outputs = train_features_batch(cfg, largest_possible_batch, model, submodule, device, dtype)
        # Include the per-example loss computation the training loop does, so its memory is covered
        per_example_token_losses(outputs.logits, largest_possible_batch.labels)
        outputs.loss.backward()
        del outputs
        dummy_optimizer.step()
        dummy_optimizer.zero_grad()

    del dummy_optimizer
    torch.cuda.empty_cache()
    gc.collect()

    print("OOM preflight check complete")


def train_model(
    cfg: SelfInterpTrainingConfig,
    training_data: list[TrainingDataPoint],
    eval_datasets: dict[str, list[TrainingDataPoint]],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    model_kwargs: dict[str, Any],
    verbose: bool = False,
):
    # Distributed settings (always on; launch with torchrun, even on 1 GPU)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Ensure loads happen on this GPU only (important for quantized models)
    model_kwargs = {
        **model_kwargs,
        "device_map": {"": f"cuda:{local_rank}"},
    }

    set_seed(cfg.seed)
    model = load_model(cfg.model_name, dtype, **model_kwargs)

    # Not every config has layer_types (e.g. GPT-NeoX), hence the default.
    if "linear_attention" in getattr(model.config.get_text_config(), "layer_types", []):
        # Linear attention models (Gated DeltaNet, e.g. Qwen3.5 / Qwen3.6) fall back to a >10x slower torch
        # implementation without these kernels. Install them with `uv sync --extra linear-attn`.
        import causal_conv1d  # noqa: F401
        import fla  # noqa: F401

    model.enable_input_require_grads()

    if cfg.gradient_checkpointing:
        model.use_cache = False
        # Reentrant matches the transformers 4.x default the paper models were trained with. The steering hook is
        # removed before loss.backward(), so the recompute of the hooked layer runs without it; non-reentrant
        # checkpointing (the transformers 5 default) errors on that. Gradients are unaffected because the hook
        # adds a detached vector.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})

    submodule = get_hf_submodule(model, cfg.hook_onto_layer)

    if cfg.resume_from is not None:
        model = PeftModel.from_pretrained(model, cfg.resume_from, is_trainable=True, autocast_adapter_dtype=True)
    elif cfg.use_lora and cfg.load_lora_path is None:
        target_modules = cfg.lora_target_modules
        vlm_targets = get_text_only_lora_targets(cfg.model_name)
        if vlm_targets and target_modules == "all-linear":
            print(f"VLM detected ({cfg.model_name}): excluding vision tower from LoRA")
            target_modules = vlm_targets

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config, autocast_adapter_dtype=True)
    elif cfg.load_lora_path is not None:
        assert cfg.resume_from is None, "Set either load_lora_path or resume_from, not both"
        load_lora_path = Path(cfg.load_lora_path)
        assert load_lora_path.exists()
        model = PeftModel.from_pretrained(model, load_lora_path, is_trainable=True, autocast_adapter_dtype=True)

    model.print_trainable_parameters()

    # Wrap with DDP for training, but keep the PEFT model reference for hooks/eval
    torch.cuda.set_device(local_rank)
    train_model_module: torch.nn.Module = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
    )

    train_model_module.train()

    oom_preflight_check(cfg, training_data, model, submodule, tokenizer, device, dtype)

    set_seed(cfg.seed)

    optimizer = torch.optim.AdamW(train_model_module.parameters(), lr=cfg.lr)

    if cfg.train_tokens_per_batch is None:
        global_step_size = cfg.train_batch_size * world_size
        effective_steps = (len(training_data) // global_step_size) * global_step_size
        if effective_steps != len(training_data):
            print(f"Trimming training_data from {len(training_data)} to {effective_steps} for equal DDP steps")
            training_data = training_data[:effective_steps]
        # Shard dataset per rank (simple strided split)
        rank_data = training_data[rank::world_size]
        rank_batches = [
            rank_data[i : i + cfg.train_batch_size] for i in range(0, len(rank_data), cfg.train_batch_size)
        ]
    else:
        batches = token_budget_batches(training_data, cfg.train_tokens_per_batch, cfg.train_batch_size)
        batches = batches[: (len(batches) // world_size) * world_size]
        training_data = [dp for batch in batches for dp in batch]
        # Shard whole batches per rank so every rank runs the same number of steps
        rank_batches = batches[rank::world_size]
        print(f"Token-budget batching: {len(batches)} batches, {len(training_data) / len(batches):.1f} rows/batch avg")

    # Token accounting (approx): count tokens after the DDP trim and before sharding.
    # This slightly overestimates actual training tokens because we later trim per-rank
    # to align with gradient_accumulation_steps.
    if rank == 0:
        tokens_per_epoch_est = sum(len(dp.input_ids) for dp in training_data)
        total_training_tokens_est = tokens_per_epoch_est * cfg.num_epochs
        num_examples_pre_shard = len(training_data)

    batches_per_epoch = (len(rank_batches) // cfg.gradient_accumulation_steps) * cfg.gradient_accumulation_steps
    if batches_per_epoch != len(rank_batches) and rank == 0:
        print(
            f"Trimming per-rank batches from {len(rank_batches)} to {batches_per_epoch} "
            "to align with gradient_accumulation_steps"
        )
    rank_batches = rank_batches[:batches_per_epoch]

    # Aggregate losses over one megabatch cycle: batches are length-grouped, and length correlates with task, so the
    # per-step loss swings with batch composition. A megabatch holds the whole task mix.
    avg_rows = sum(len(b) for b in rank_batches) / len(rank_batches)
    log_window = max(1, round(cfg.window_mult * cfg.train_batch_size / avg_rows))
    window_losses: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    # A fixed sample of training examples (two per task) whose oracle answers are logged as training goes on
    example_points: list[TrainingDataPoint] = []
    for dp in training_data:
        if sum(task_group(e.datapoint_type) == task_group(dp.datapoint_type) for e in example_points) < 2:
            example_points.append(dp)

    steps_per_epoch = batches_per_epoch // cfg.gradient_accumulation_steps
    assert steps_per_epoch > 0, "No optimizer steps will be run; check dataset/batch/accumulation sizes"
    total_training_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_training_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    # --------------------------------------------------------------

    global_step = 0
    resume_epoch = 0
    resume_step_idx = 0
    if cfg.resume_from is not None:
        trainer_state = torch.load(Path(cfg.resume_from) / "trainer_state.pt", map_location=device, weights_only=False)
        optimizer.load_state_dict(trainer_state["optimizer"])
        scheduler.load_state_dict(trainer_state["scheduler"])
        global_step = trainer_state["global_step"]
        resume_epoch = trainer_state["epoch"]
        resume_step_idx = trainer_state["next_step_idx"]
        print(f"Resuming from {cfg.resume_from}: global_step {global_step}, epoch {resume_epoch}, step_idx {resume_step_idx}")

    # Init Weights & Biases only on rank 0
    if rank == 0:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=asdict(cfg))
        wandb.summary["train/tokens_per_epoch_est"] = tokens_per_epoch_est
        wandb.summary["train/total_tokens_est"] = total_training_tokens_est
        wandb.summary["train/num_examples_pre_shard"] = num_examples_pre_shard

    train_start_time = time.time()
    padded_tokens_trained = 0
    for epoch in range(resume_epoch, cfg.num_epochs):
        accumulated_loss = 0.0
        optimizer.zero_grad()
        for step_idx, t_batch_list in enumerate(
            tqdm(rank_batches, desc=f"Training epoch {epoch + 1}", disable=rank != 0)
        ):
            if epoch == resume_epoch and step_idx < resume_step_idx:
                continue

            # Compute missing steering vectors using the PEFT model (not DDP wrapper)
            t_batch_list = materialize_missing_steering_vectors(t_batch_list, tokenizer, model)

            t_batch = construct_batch(t_batch_list, tokenizer, device)
            padded_tokens_trained += t_batch.input_ids.numel()

            # Forward/backward on the DDP-wrapped module if enabled
            outputs = train_features_batch(cfg, t_batch, train_model_module, submodule, device, dtype)
            loss_sums, token_counts = per_example_token_losses(outputs.logits, t_batch.labels)
            for dp, loss_sum, count in zip(t_batch_list, loss_sums.tolist(), token_counts.tolist(), strict=True):
                for key in ("train/loss_megabatch", f"train_loss_by_task/{task_group(dp.datapoint_type)}",
                            f"train_loss_by_dataset/{dp.datapoint_type}"):
                    window_losses[key][0] += loss_sum
                    window_losses[key][1] += count
            loss = outputs.loss
            del outputs
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

            is_update_step = (step_idx + 1) % cfg.gradient_accumulation_steps == 0

            if is_update_step:
                grad_norm = clip_grad_norm_(train_model_module.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if rank == 0:
                    elapsed = time.time() - train_start_time
                    log = {
                        "train/loss": accumulated_loss,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/grad_norm": grad_norm.item(),
                        "train/batch_rows": len(t_batch_list),
                        "train/batch_padded_tokens": t_batch.input_ids.numel(),
                        "train/padded_tokens_per_sec": padded_tokens_trained / elapsed,
                        # Peak memory of this step, including activation collection: how well the batch fills the GPU
                        "train/peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                        "train/peak_mem_frac": torch.cuda.max_memory_allocated()
                        / torch.cuda.get_device_properties(device).total_memory,
                    }
                    torch.cuda.reset_peak_memory_stats()
                    if (global_step + 1) % log_window == 0:
                        # Token-weighted mean loss over the last megabatch, overall and per task
                        log.update({k: loss_sum / count for k, (loss_sum, count) in window_losses.items()})
                        window_losses.clear()
                    if global_step % cfg.example_log_steps == 0:
                        log["train_examples"] = training_examples_table(
                            cfg, example_points, model, tokenizer, submodule, device, dtype, global_step
                        )
                    wandb.log(log, step=global_step)
                    if verbose:
                        print(f"Step {global_step} loss: {accumulated_loss}")
                    if global_step % 50 == 0:
                        elapsed = time.time() - train_start_time
                        print(f"Throughput at step {global_step}: {padded_tokens_trained / elapsed:.0f} padded tokens/s")

                # -------------------------------- evaluation --------------------------------
                if global_step % cfg.eval_steps == 0 and (cfg.eval_on_start or global_step > 0):
                    if rank == 0:
                        eval_all_datasets(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step)
                    dist.barrier()

                if global_step % cfg.save_steps == 0 and global_step > 0:
                    if rank == 0:
                        ckpt_dir = f"{cfg.save_dir}/step_{global_step}"
                        model.save_pretrained(ckpt_dir)
                        torch.save(
                            {
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "global_step": global_step + 1,
                                "epoch": epoch,
                                "next_step_idx": step_idx + 1,
                            },
                            f"{ckpt_dir}/trainer_state.pt",
                        )
                        if cfg.hf_push_to_hub and cfg.hf_repo_id:
                            print("Pushing LoRA adapter to Hugging Face Hub...")
                            push_lora_to_hf(
                                model=model,
                                tokenizer=tokenizer,
                                repo_id=cfg.hf_repo_id + f"-step-{global_step}",
                                private=cfg.hf_private_repo,
                                commit_message=(f"SAE introspection LoRA - {cfg.wandb_run_name} - step {global_step}"),
                            )
                            print("Pushed LoRA adapter to Hugging Face Hub.")
                    dist.barrier()

                global_step += 1
                accumulated_loss = 0.0

    train_time = time.time() - train_start_time
    print(f"Training complete in {train_time:.0f}s, {padded_tokens_trained / train_time:.0f} padded tokens/s")

    # Save final model
    if rank == 0:
        print("Saving final model...")
        model.save_pretrained(f"{cfg.save_dir}/final")

        # Final evaluation
        print("Running final evaluation...")
        eval_all_datasets(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step)
        wandb.finish()

        # Push to Hugging Face if configured
        if cfg.hf_push_to_hub and cfg.hf_repo_id:
            print("Pushing LoRA adapter to Hugging Face Hub...")
            push_lora_to_hf(
                model=model,
                tokenizer=tokenizer,
                repo_id=cfg.hf_repo_id,
                commit_message=f"SAE introspection LoRA - {cfg.wandb_run_name} - final model",
                private=cfg.hf_private_repo,
            )
    dist.barrier()


def length_grouped_reorder(
    data: list[TrainingDataPoint],
    batch_size: int,
    window_mult: int,
) -> list[TrainingDataPoint]:
    lengths = [len(d.input_ids) for d in data]

    indices = list(range(len(data)))
    megabatch_size = window_mult * batch_size

    # Slice into mega-batches
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(indices), megabatch_size)]
    # Sort within each mega-batch by length desc
    megabatches = [sorted(mb, key=lambda i: lengths[i], reverse=True) for mb in megabatches]

    new_order = [i for mb in megabatches for i in mb]
    return [data[i] for i in new_order]


def token_budget_batches(
    data: list[TrainingDataPoint],
    max_tokens: int,
    max_rows: int,
) -> list[list[TrainingDataPoint]]:
    """Greedily chunks (length-grouped) data into batches of at most max_tokens padded tokens and max_rows rows.
    Padded tokens = rows * longest row, since construct_batch pads every row to the longest."""
    batches = []
    batch = []
    batch_max_len = 0
    for dp in data:
        assert len(dp.input_ids) <= max_tokens, f"Datapoint of {len(dp.input_ids)} tokens exceeds max_tokens"
        new_max_len = max(batch_max_len, len(dp.input_ids))
        if batch and (new_max_len * (len(batch) + 1) > max_tokens or len(batch) == max_rows):
            batches.append(batch)
            batch = []
            new_max_len = len(dp.input_ids)
        batch.append(dp)
        batch_max_len = new_max_len
    batches.append(batch)
    return batches


def build_datasets(
    cfg: SelfInterpTrainingConfig,
    dataset_loaders: list[ActDatasetLoader],
    max_len_percentile: float | None = 0.999,
    window_mult: int | None = 20,
) -> tuple[list[TrainingDataPoint], dict[str, list[TrainingDataPoint]]]:
    set_seed(cfg.seed)
    all_training_data: list[TrainingDataPoint] = []
    # eval data will only be for classification datasets
    all_eval_data: dict[str, list[TrainingDataPoint]] = {}

    for dataset_loader in dataset_loaders:
        if "train" in dataset_loader.dataset_config.splits:
            all_training_data.extend(dataset_loader.load_dataset("train"))
        if "test" in dataset_loader.dataset_config.splits:
            all_eval_data[dataset_loader.dataset_config.dataset_name] = dataset_loader.load_dataset("test")

    p = max_len_percentile
    if p is not None:
        if p >= 1.0 or p <= 0.0:
            raise ValueError("max_len_percentile must be less than 1.0 and greater than 0.0")

        lengths = sorted(len(td.input_ids) for td in all_training_data)
        median_length = lengths[len(lengths) // 2]
        print(f"Max length: {lengths[-1]}, Min length: {lengths[0]}, Median length: {median_length}")
        # Inclusive quantile index
        idx = int((len(lengths) - 1) * p)
        threshold = lengths[idx]

        before = len(all_training_data)
        all_training_data = [td for td in all_training_data if len(td.input_ids) <= threshold]
        removed = before - len(all_training_data)
        print(f"Percentile trim: kept <= {threshold} tokens (p={p:.6f}). Removed {removed}/{before} examples.")

    set_seed(cfg.seed)
    random.shuffle(all_training_data)

    if window_mult is not None:
        all_training_data = length_grouped_reorder(all_training_data, cfg.train_batch_size, window_mult)

    return all_training_data, all_eval_data


# Helper to cut repetition when building DatasetLoaderConfig
def mk_cfg(
    custom_params,
    *,
    num_train: int,
    num_test: int,
    splits: list[str],
    model_name: str,
    layer_percents: list[int],
    save_acts: bool,
    batch_size: int,
) -> DatasetLoaderConfig:
    return DatasetLoaderConfig(
        custom_dataset_params=custom_params,
        num_train=num_train,
        num_test=num_test,
        splits=splits,
        model_name=model_name,
        layer_percents=layer_percents,
        save_acts=save_acts,
        batch_size=batch_size,
    )


def build_loader_groups(
    *,
    model_name: str,
    layer_percents: list[int],
    act_collection_batch_size: int,
    save_acts: bool,
    classification_datasets: dict[str, dict[str, Any]],
    model_kwargs: dict[str, Any],
) -> dict[str, list[ActDatasetLoader]]:
    DEBUG = False
    num_datapoints = 10_000  # 10x scaled down from 100_000

    # DEBUG = True

    if DEBUG:
        print("DEBUG mode: using small datasets")
        num_datapoints = 100

    # PastLens: build both single-token and multi-token variants
    past_lens_single = PastLensDatasetLoader(
        dataset_config=mk_cfg(
            PastLensDatasetConfig(
                max_k_activations=1,
                max_k_tokens=50,
            ),
            num_train=num_datapoints,
            num_test=0,
            splits=["train"],
            model_name=model_name,
            layer_percents=layer_percents,
            save_acts=save_acts,
            batch_size=train_batch_size,
        )
    )

    past_lens_multi = PastLensDatasetLoader(
        dataset_config=mk_cfg(
            PastLensDatasetConfig(
                max_k_activations=50,
                max_k_tokens=50,
            ),
            num_train=num_datapoints,
            num_test=0,
            splits=["train"],
            model_name=model_name,
            layer_percents=layer_percents,
            save_acts=save_acts,
            batch_size=train_batch_size,
        )
    )

    latent_qa_loader = LatentQADatasetLoader(
        dataset_config=mk_cfg(
            custom_params=LatentQADatasetConfig(),
            num_train=6_500,  # 10x scaled down from the full ~65k LatentQA train set (100_000 used all of it)
            num_test=0,
            splits=["train"],
            model_name=model_name,
            layer_percents=layer_percents,
            save_acts=False,
            batch_size=train_batch_size,
        )
    )

    # SAE datasets per layer percent
    sae_loaders: list[ActDatasetLoader] = []
    sae_explanation_loaders: list[ActDatasetLoader] = []
    for layer_percent in layer_percents:
        sft_data_path = (
            f"sae_data/qwen_hard_negatives_0_20000_layer_percent_{layer_percent}_sft_data_gpt-5-mini-2025-08-07.jsonl"
        )

        sae_explanation_loaders.append(
            SAEExplanationDatasetLoader(
                dataset_config=mk_cfg(
                    SAEExplanationDatasetConfig(
                        sft_data_file=sft_data_path,
                        use_decoder_vectors=True,
                    ),
                    num_train=20000,
                    num_test=0,
                    splits=["train"],
                    model_name=model_name,
                    layer_percents=[layer_percent],
                    save_acts=True,
                    batch_size=0,
                )
            )
        )

        sae_loaders.append(
            SAEActivatingSequencesDatasetLoader(
                dataset_config=mk_cfg(
                    SAEActivatingSequencesDatasetConfig(
                        sae_repo_id="adamkarvonen/qwen3-8b-saes",
                        use_decoder_vectors=True,
                    ),
                    num_train=60000,
                    num_test=0,
                    splits=["train"],
                    model_name=model_name,
                    layer_percents=[layer_percent],
                    save_acts=True,
                    batch_size=0,
                )
            )
        )

        sae_loaders.append(
            SAEYesNoDatasetLoader(
                dataset_config=mk_cfg(
                    SAEYesNoDatasetConfig(sft_data_file=sft_data_path),
                    num_train=60000,
                    num_test=0,
                    splits=["train"],
                    model_name=model_name,
                    layer_percents=[layer_percent],
                    save_acts=True,
                    batch_size=0,
                )
            )
        )

    # Classification: build both single-token and multi-token variants for each dataset
    classification_loaders: list[ActDatasetLoader] = []
    for ds_name, meta in classification_datasets.items():
        single_params = ClassificationDatasetConfig(
            classification_dataset_name=ds_name,
            max_window_size=1,
            min_end_offset=-1,
            max_end_offset=-5,
            num_qa_per_sample=2,
        )
        multi_params = ClassificationDatasetConfig(
            classification_dataset_name=ds_name,
            max_window_size=50,
            min_end_offset=-1,
            max_end_offset=-5,
            num_qa_per_sample=1,
        )

        # language identification has very long sequence lengths
        if "batch_size" in meta:
            bs = meta["batch_size"]
        else:
            bs = train_batch_size

        classification_loaders.append(
            ClassificationDatasetLoader(
                dataset_config=mk_cfg(
                    single_params,
                    num_train=meta["num_train"],
                    num_test=meta["num_test"],
                    splits=meta["splits"],
                    model_name=model_name,
                    layer_percents=layer_percents,
                    save_acts=save_acts,
                    batch_size=bs,
                ),
                model_kwargs=model_kwargs,
            )
        )

        classification_loaders.append(
            ClassificationDatasetLoader(
                dataset_config=mk_cfg(
                    multi_params,
                    num_train=meta["num_train"],
                    num_test=meta["num_test"],
                    splits=meta["splits"],
                    model_name=model_name,
                    layer_percents=layer_percents,
                    save_acts=save_acts,
                    batch_size=train_batch_size,
                ),
                model_kwargs=model_kwargs,
            )
        )

    return {
        "past_lens_loaders": [past_lens_single, past_lens_multi],
        "latentqa_loaders": [latent_qa_loader],
        "classification_loaders": classification_loaders,
        "sae_loaders": sae_loaders,
        "sae_explanation_loaders": sae_explanation_loaders,
    }


def _ensure_datasets_exist(dataset_loaders: list[ActDatasetLoader]) -> None:
    """Materialize datasets on disk using a single process (rank 0).

    Each loader's `load_dataset` will create and save if missing; otherwise it
    simply loads. This avoids race conditions when multiple ranks start up.
    """

    # TODO: Switch to multiprocessing for speed

    old_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    # Make only GPU 0 visible for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        for dl in dataset_loaders:
            for split in dl.dataset_config.splits:
                _ = dl.load_dataset(split)
    finally:
        # Revert to original state
        if old_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible_devices


if __name__ == "__main__":
    # for gemma: export TORCHDYNAMO_DISABLE=1
    # Always initialize DDP (launch with torchrun, even for 1 GPU)
    # time delta of two hours because currently it can take 1 hour to build all datasets
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()

    main_train_size = 600  # 10x scaled down from 6000
    main_test_size = 250
    classification_datasets = {
        "geometry_of_truth": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "relations": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "sst2": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "md_gender": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "snli": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "ag_news": {"num_train": main_train_size, "num_test": main_test_size, "splits": ["test"]},
        "ner": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "tense": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "language_identification": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["test"],
            # language identification has very long sequence lengths
            "batch_size": 4,
        },
        "singular_plural": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
    }

    dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")

    hook_layer = 1
    # model_name = "Qwen/Qwen3-32B"
    # model_name = "meta-llama/Llama-3.3-70B-Instruct"
    # model_name = "google/gemma-2-9b-it"
    # model_name = "Qwen/Qwen3-8B"

    models = [
        # "Qwen/Qwen3-14B",
        # "google/gemma-2-27b-it",
        # "meta-llama/Llama-3.1-8B-Instruct",
        # "google/gemma-3-4b-it",
        # "google/gemma-3-12b-it",
        # "google/gemma-3-27b-it",
        # "Qwen/Qwen3-4B",
        # "Qwen/Qwen3.6-27B",
        "Qwen/Qwen3.5-0.8B",
    ]

    for model_name in models:
        hf_repo_name = "N/A"

        model_name_str = model_name.split("/")[-1].replace(".", "_").replace(" ", "_")

        train_batch_size = 16
        train_tokens_per_batch = None
        eval_batch_size = train_batch_size * 8
        if model_name == "Qwen/Qwen3.5-0.8B":
            # Token-budget batching: fill the GPU on every step instead of sizing batches for the longest prompt.
            # train_batch_size caps rows (each row also needs a context forward pass to collect activations).
            train_batch_size = 256
            train_tokens_per_batch = 8192
            eval_batch_size = 512
        # Scale LR linearly with global batch size from the 1e-5 @ batch 16 reference. With token-budget batching
        # this is overwritten below using the average rows per batch.
        lr = 1e-5 * train_batch_size / 16
        # Small models fit in memory without gradient checkpointing, and skipping the recompute is faster
        gradient_checkpointing = model_name not in {"Qwen/Qwen3.5-0.8B"}
        model_kwargs = {}
        save_steps = 5_000

        if model_name == "Qwen/Qwen3-32B" or model_name == "meta-llama/Llama-3.3-70B-Instruct":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=dtype,
            )
            model_kwargs = {"quantization_config": bnb_config}

        # if model_name == "meta-llama/Llama-3.3-70B-Instruct":
        # train_batch_size = train_batch_size * 4  # increase gpu utilization on 4x GPUs
        # cuts training time by ~50%

        print("Global train batch size:", train_batch_size)
        assert train_batch_size % world_size == 0, (
            f"Global batch size {train_batch_size} must be divisible by world_size {world_size}"
        )
        train_batch_size = train_batch_size // world_size
        print(f"Per-rank train batch size: {train_batch_size}, world size: {world_size}")

        layer_percents = [25, 50, 75]
        save_acts = False

        gradient_accumulation_steps = 1

        # Build loader groups (single + multi variants)
        loader_groups = build_loader_groups(
            model_name=model_name,
            layer_percents=layer_percents,
            act_collection_batch_size=train_batch_size,
            save_acts=save_acts,
            classification_datasets=classification_datasets,
            model_kwargs=model_kwargs,
        )

        classification_dataset_loaders = loader_groups["classification_loaders"]
        past_lens_loaders = loader_groups["past_lens_loaders"]
        sae_dataset_loaders = loader_groups["sae_loaders"]
        sae_explanation_dataset_loaders = loader_groups["sae_explanation_loaders"]
        latentqa_loaders = loader_groups["latentqa_loaders"]

        iterations = [
            # Default dataset mixture
            # Set load_lora_path to checkpoint path to continue training
            {
                "load_lora_path": None,
                "resume_from": os.environ.get("AO_RESUME_FROM"),
                "dataset_loaders": latentqa_loaders + classification_dataset_loaders + past_lens_loaders,
                "wandb_suffix": f"_latentqa_cls_past_lens_{model_name_str}",
            },
            # {
            #     "load_lora_path": None,
            #     "dataset_loaders": latentqa_loaders,
            #     "wandb_suffix": f"_latentqa_only_{model_name_str}",
            # },
        ]

        for hyperparam_override in iterations:
            loop_dataset_loaders = hyperparam_override.pop("dataset_loaders")
            if hyperparam_override["load_lora_path"] is not None:
                assert os.path.exists(hyperparam_override["load_lora_path"]), f"{hyperparam_override['load_lora_path']}"

            cfg = SelfInterpTrainingConfig(
                model_name=model_name,
                hook_onto_layer=hook_layer,
                hf_repo_name=hf_repo_name,
                # wandb_suffix=wandb_suffix,
                layer_percents=layer_percents,
                train_batch_size=train_batch_size,
                train_tokens_per_batch=train_tokens_per_batch,
                lr=lr,
                activation_collection_batch_size=train_batch_size * 4,
                eval_batch_size=eval_batch_size,
                eval_steps=10_000,
                eval_on_start=True,
                gradient_checkpointing=gradient_checkpointing,
                gradient_accumulation_steps=gradient_accumulation_steps,
                save_steps=save_steps,
                **hyperparam_override,
            )

            cfg.finalize(dataset_loaders=loop_dataset_loaders)

            print(f"save dir: {cfg.save_dir}")

            tokenizer = load_tokenizer(cfg.model_name)

            # Ensure only rank 0 performs any on-disk dataset creation
            if local_rank == 0:
                _ensure_datasets_exist(loop_dataset_loaders)
            dist.barrier()

            all_training_data, all_eval_data = build_datasets(
                cfg, dataset_loaders=loop_dataset_loaders, window_mult=cfg.window_mult
            )

            if cfg.train_tokens_per_batch is not None:
                num_batches = len(
                    token_budget_batches(all_training_data, cfg.train_tokens_per_batch, cfg.train_batch_size)
                )
                avg_global_rows = len(all_training_data) / num_batches * world_size
                cfg.lr = 1e-5 * avg_global_rows / 16
                print(f"Token-budget batching: {avg_global_rows:.1f} global rows/batch avg, lr {cfg.lr:.2e}")

            # for debugging
            # all_training_data = all_training_data[:100]
            # eval_keys = list(all_eval_data.keys())
            # assert len(eval_keys) == 1
            # eval_key = eval_keys[0]
            # all_eval_data = {eval_key: all_training_data[:]}

            print(f"training data length: {len(all_training_data)}, eval data length: {len(all_eval_data)}")

            print(asdict(cfg))

            train_model(
                cfg=cfg,
                training_data=all_training_data,
                eval_datasets=all_eval_data,
                tokenizer=tokenizer,
                dtype=dtype,
                device=device,
                model_kwargs=model_kwargs,
                verbose=True,
            )

    # Clean up DDP
    dist.destroy_process_group()
