# Improve activation oracle training

An activation oracle is an LLM fine-tuned (with a LoRA adapter) to read another model's activations and answer
natural-language questions about them. Your job is to find a better training setup for an activation oracle on
**Qwen/Qwen3.5-0.8B**, working autonomously like a researcher: form hypotheses, run experiments, keep what helps.

## What you're scored on

Three tiers of tasks. You can score the first two (through eval tools, below); the third is hidden.

1. **Dev metrics** (hill-climb on these):
   - **Taboo.** 20 target LoRAs were each fine-tuned to play a word game: give hints about one secret word without
     ever saying it. The oracle sees activations of a user message to the target and must name the secret word.
     Baseline: about 21%.
   - **ag_news.** Yes/no questions about the topic of a news snippet, a classification dataset that isn't in the
     training mixture. Baseline: about 67% (chance 50%).
2. **Validation metrics** (confirm that improvements generalize; don't climb them):
   - **PersonaQA.** A target LoRA was fine-tuned on biographies of 100 fictional people. The oracle sees activations
     of `"My name is {name}."` and must recover attributes such as the person's country or favorite food. Baseline:
     about 5%, and it moves slowly.
   - **OOD classification.** Two more yes/no classification datasets that aren't in the training mixture (which
     language a text is in; whether a noun is singular or plural). Baseline: about 64% (chance 50%).
3. **Held-out test tasks** that you cannot see, testing other things an oracle should be able to read from
   activations. **These are the main score.** Your submission is retrained from scratch in a fresh sandbox and scored
   on them, so prefer changes that make the oracle generally better over changes that only raise Taboo.

None of these are training data: the oracle trains on the mixture in `nl_probes/sft.py` (LatentQA, classification and
PastLens data), which you can change.

## Your environment

- `/app/repo`: the codebase (a git repo with the baseline committed). Read `nl_probes/sft.py` (training entry point
  and data mixture), `nl_probes/dataset_classes/` (data generators), `nl_probes/utils/steering_hooks.py` (how
  activations are injected) and `nl_probes/base_experiment.py` (inference).
- One A100 80GB GPU.
- No internet. The base model, classification datasets and pretraining/chat text slices (`/opt/ao_data/`) are
  preloaded; the HuggingFace hub is unreachable, so you can't download anything new. The one exception is wandb:
  `nl_probes/sft.py` runs log to the `activation_oracles` project automatically. Give each experiment a short
  descriptive run name (`wandb_suffix` in `nl_probes/sft.py`) so runs are easy to compare.

## Scoring an oracle

The eval code, target models and eval data are not in your sandbox. You score an oracle through two MCP tools (server
`ao-eval`), after copying its adapter into `/shared`:

```bash
cd /app/repo
bash train.sh                       # trains and writes the oracle LoRA to oracle_out/
cp -r oracle_out /shared/baseline   # then call evaluate_dev(adapter_name="baseline")
```

| Tool | Returns | Budget |
| - | - | - |
| `evaluate_dev` | Taboo and ag_news accuracy (the dev metrics), a couple of minutes | 100 calls |
| `evaluate_val` | PersonaQA and OOD classification accuracy (the validation metrics), a few minutes | **10 calls** |

Calls that fail (for example a malformed adapter) still count. The verifier scores Taboo on a different set of
prompts from the ones the tool uses, so don't tune to tiny Taboo differences.

## Rules

- **Submission:** the final state of `/app/repo`. The verifier runs `bash train.sh` with `OUTPUT_DIR` set, and
  `train.sh` must write a PEFT LoRA adapter for `Qwen/Qwen3.5-0.8B` to `$OUTPUT_DIR`. Caches, checkpoints and weight
  files (`*.pt`, `*.safetensors`, `*.bin`, `*.npy`, `sft_training_data/`, `checkpoints*/`) are not submitted, so
  `train.sh` must train from scratch.
- **Time limit for `train.sh`:** at most 40 minutes on one A100 80GB, including dataset building. The verifier kills it
  at 50 minutes and scores 0. The baseline takes about 33 minutes (about 8 of them building datasets).
- **Stay compatible with the reference inference code.** Evaluation uses an unmodified copy of the repo
  (`/opt/ao_ref`), not your copy. It loads your adapter onto Qwen/Qwen3.5-0.8B, collects activations from layer 12
  (50% depth) at the last 10 context tokens (which end with the chat template's assistant header), and injects them at
  layer 1 with the standard additive hook and the `"Layer: {L}\n" + " ?" * N` prompt prefix. Answers are scored by
  whether they contain the correct value. You can change anything about *training* (data, mixture, losses,
  hyperparameters, new data generators, layers you train on), but the trained adapter must work with that inference
  path.
- **Eval tasks are metrics, not training data.** Don't build training data that recreates the Taboo, PersonaQA or
  OOD classification tasks (for example a word game around secret words, or questions about fictional people's
  favorite foods). Otherwise the scores stop telling you anything.
- Don't modify anything under `/opt`.

## How to work

- You have **2.5 hours** in total, and the session ends hard at the limit. Run `time_left` to check your budget. You
  can't end the session before the last 15 minutes, so plan to use the whole time.
- A full baseline run takes a large share of the time budget, so iterate with smaller proxy runs (less data, fewer
  steps) and confirm promising changes with full runs. Validation calls are scarce: save them for changes worth confirming.
- Commit every improvement, so `/app/repo` is always a valid submission. If time runs out mid-experiment, the last
  committed state should be what's there: revert uncommitted changes that aren't better.
- Keep a log of experiments and results in `/app/repo/NOTES.md`.
- When you finish, make sure no training or eval processes are still running.
