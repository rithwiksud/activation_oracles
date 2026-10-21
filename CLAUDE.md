# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working rules (from AGENTS.md)

- Do not change any code unless explicitly asked. By default, give suggestions: ideas and code snippets the user can integrate.
- Prefer code that fails loudly. Avoid defensive guards like `try/except`, `dict.get()`, `if len(data) == 0` unless clearly needed. No silent failures, no bloat.

## Project

Code for the Activation Oracles paper (arXiv 2512.15674): LLMs fine-tuned (via LoRA) to take another model's activations as input and answer natural-language questions about them. Package name is `nl_probes`.

`main` tracks transformers 5 / torch 2.12 and newer models (e.g. Qwen3.6-27B). Exact paper reproduction lives on the `paper` branch / `v1.0-paper` tag (transformers 4.55.2, torch 2.7.1, peft 0.17.1), so don't "fix" `main` to match paper numbers.

## Commands

```bash
uv sync                                  # install (add --extra linear-attn for Qwen3.5/3.6 Gated DeltaNet training kernels)
source .venv/bin/activate

torchrun --nproc_per_node=<NUM_GPUS> nl_probes/sft.py   # train; always use torchrun, even on 1 GPU (DDP is always initialized)
AO_RESUME_FROM=<save_dir>/step_<N> torchrun ...          # resume from a checkpoint (adapter + optimizer state)

bash experiments/paper_evals.sh          # classification, gender, taboo, SSC, PersonaQA evals
python experiments/taboo_open_ended_eval.py              # run a single eval

pytest tests/                            # tests download HF configs/tokenizers/models (some gated)
pytest tests/test_positions.py -k Qwen   # single test
```

There is no lint/format config in the repo. Code style is ~120-char lines.

Scripts are configured by editing their `if __name__ == "__main__":` block (model list, layer percents, dataset sizes, LoRA paths), not by CLI flags. For training, the model list is at the bottom of `nl_probes/sft.py`; defaults live in `nl_probes/configs/sft_config.py`. `vllm` is not a dependency. The few scripts that use it (patchscopes, personaqa knowledge evals) need a separate env.

## Architecture

**Activation injection (the core mechanism).** An oracle prompt starts with `get_introspection_prefix` (`nl_probes/utils/dataset_utils.py`): `"Layer: {L}\n"` followed by one `" ?"` placeholder token per injected activation. During the forward pass, `get_hf_activation_steering_hook` (`utils/steering_hooks.py`) is registered on decoder layer `hook_onto_layer` / `injection_layer` (default 1). At each placeholder position it **adds** `normalize(v) * ||resid|| * steering_coefficient` to the residual. The hook only fires when seq len > 1 (prompt pass, not cached decode steps). `get_hf_submodule` (`utils/activation_utils.py`) resolves layers structurally via `model.get_decoder().layers[i]`, which works for both CausalLM and VLM wrappers (Gemma-3, Qwen3.6).

**Layers.** Source activations come from layers given as percents of depth (default 25/50/75), converted by `layer_percent_to_layer` using `AutoConfig...get_text_config().num_hidden_layers`.

**Data model.** Everything is a `TrainingDataPoint` (pydantic, `extra="forbid"`). A point either has precomputed `steering_vectors` or has `steering_vectors=None` plus `context_input_ids` / `context_positions`. In the second case, `materialize_missing_steering_vectors` computes the activations lazily per batch, from the *same* model with the LoRA adapter disabled (`model.disable_adapter()`), in one forward pass with hooks on all requested layers. `save_acts=False` (the default in `sft.py`) means activations are generated on the fly this way. Batches are left-padded (`construct_batch`), and positions are shifted accordingly.

**Datasets.** `dataset_classes/` holds `ActDatasetLoader` subclasses: classification (`classification_dataset_manager.py`, reading `datasets/classification_datasets/`), LatentQA (system-prompt QA), and PastLens (self-supervised context prediction). `load_dataset` builds the dataset on first use and caches it as a `.pt` file in `dataset_folder` (default `sft_training_data/`). The filename includes model, size, split, and a hash of the config, so changing config regenerates it. In `sft.py`, `build_loader_groups` assembles the training mixture and `_ensure_datasets_exist` builds caches on a single process before DDP training.

**Training** (`nl_probes/sft.py`). Custom DDP loop (not HF Trainer): LoRA on the base model, length-grouped batching, OOM preflight on the longest prompts, periodic eval via `utils/eval.py`, checkpoint saving, optional HF Hub push. Gemma-3 needs text-only LoRA target patterns (`get_text_only_lora_targets`) so vision-tower params don't break DDP. Linear-attention models require `fla` at training time (checked at startup).

**Evaluation** (`nl_probes/base_experiment.py`). Shared harness used by `experiments/*_eval.py`. `VerbalizerEvalConfig` controls which activation types are fed to the oracle (`orig` = base model, `lora` = target fine-tune such as a taboo/secret-keeping LoRA, `diff` = lora − orig) and which positions (`tokens` = per-token, `segment`, `full_seq`). The flow is: `collect_target_activations` (toggling adapters on one PEFT model holding both target and oracle LoRAs) → `create_verbalizer_inputs` → `run_verbalizer`. Results go to JSON. Plotting scripts live in `experiments/plotting/` and `experiments/final_paper_plots/`.

`experiments/activation_oracle_demo.ipynb` is a self-contained inference example (no `nl_probes` imports). Keep it that way.
