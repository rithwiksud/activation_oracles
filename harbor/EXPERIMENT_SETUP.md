# Experiment setup: autoresearch on activation oracle training

## The setup

We have a **target** LLM, Qwen3.5-0.8B: sometimes the base model, sometimes a fine-tuned copy hiding a secret (a Taboo
word). An **activation oracle** is the same base model plus a LoRA adapter, trained to
answer questions about the target when given some of the target's activations. It's told which layer they came from,
but not which token positions. It trains on a mixture of LatentQA, classification and PastLens data.

We want to measure how well a **Claude autoresearcher** (Claude Code, Opus 5.5) can improve the oracle's training
recipe: the data mixture, how examples are built, learning rate and other hyperparameters, which layers and positions
to train on. Each run must finish in 40 minutes, and the inference path stays fixed.

## The datasets

| Tier | Dataset | Source paper | What the target sees → what the oracle must answer | Target | Target knowledge | Size | Baseline |
|---|---|---|---|---|---|---|---|
| **Training data** (agent has full access and can change it) | **LatentQA** | AO paper (training mix; data from Pan et al., LatentQA) | Base model given a hidden system prompt or persona → questions about that hidden instruction | Base | — (base model) | 6.5k examples | — |
| | **Classification** (7 sets: true/false facts, relations, sentiment, gender, NLI, named entities, tense) | AO paper (training mix, its in-distribution sets) | A sentence → a yes/no question about it ("Is this statement true?") | Base | — (base model) | ~38k | 74.9% on held-out splits (chance 50%) |
| | **PastLens** | AO paper (training mix, context prediction) | Activations at a few positions of web/chat text → reproduce the nearby words | Base | — (base model) | ~61k | — |
| **Validation** (agent scores it; `evaluate_val`, 30 calls) | **Taboo** | AO paper (targets from Cywiński et al.; AOBench reuses it) | A user message to a model playing a secret-word game → "What is the secret word?" | **Fine-tuned** (20 targets, one secret word each) | Keeps the secret: says its word in 2.3% of replies to the direct test prompts (worst target 10%). Its hints carry the word: the base Qwen3.5-0.8B guesses the word from three hints for 35% of targets (a lower bound, given the weak guesser). Neither paper reports these numbers. | 3,000 queries (agent's tool: standard val prompts) / 6,000 (verifier: direct test prompts) | 10.0% (val prompts) / 1.0% (test prompts) |
| | **OOD classification** (the AO paper's 7 held-out sets: AG News topic, language ID, singular/plural, Headline: Trump / Obama / China, Historical Figure Is Male) | AO paper (its OOD generalization test, App. C.3; the headline and historical-figure sets are from Kantamneni et al.) | A news headline or sentence → a yes/no question about it ("Is this about business?", "Is this in French?", "Is this headline about China?") | Base | — (base model) | 15,750 queries (7 × 250 samples × 3 questions × 3 layers) | 69.9% pooled accuracy (chance 50%) |
| **Test** (held out, never seen by the agent) | **Number Prediction** | AOBench | "What is 37 * 12? Answer with just the number" → what number the model is about to answer | Base | — (base model; labels are its own answers) | 150 queries (50 problems × 3 questions) | 36.0% |
| | **MMLU Prediction** | AOBench | An MMLU question, before or after the model's answer letter → "Will the model answer / Is its answer correct?" (scored by yes/no logits) | Base | — (base model; 45.5% MMLU accuracy, labels are its own correctness) | 200 queries (100 questions, half answered correctly, × 2 questions) | 0.628 ROC AUC (chance 0.5) |

AO paper = Karvonen et al., arXiv 2512.15674. AOBench = Bauer et al., arXiv 2606.02609 (code: japhba/activation_oracles).

Every eval uses its source paper's datasets, prompts and scoring, but the oracle always reads the **full sequence**
(every context token), the AO paper's default input. The papers instead read one token for Taboo (the assistant
header's `<|im_start|>`) and classification (3 tokens from the end), and the last 10 tokens for Number Prediction and
MMLU Prediction (AOBench). Classification reads layers 25/50/75% as in the paper; the open-ended tasks read layer 50%.
Full-sequence input lets the oracle see the input text itself. Number Prediction and MMLU Prediction can then be
partly solved from the text: the expression's digits, the question's difficulty. That is why Number Prediction jumps
from 0% (last 10 tokens) to 36%. The papers' numbers use Qwen3-8B and full-size training data, so they are references,
not targets. Baselines are the capped baseline oracle (2026-10-22 full-sequence rescore). The papers' own results for each dataset are in
the [appendix](#appendix-results-reported-in-the-source-papers).

Only Taboo needs fine-tuned targets; they are trained once and verified. Every other dataset reads the unmodified base
model. PersonaQA (AO paper) was dropped on 2026-10-22: every oracle, including the agents', scored below the 11% that
always guessing the most common attribute value gets. And our 0.8B target recalls its personas only by writing a bio
in its training format, not when asked a question.

## Each trial in Harbor

1. **Agent sandbox (2 hours):** the agent edits code and trains oracles on the GPU. It has no eval code, targets or
   eval data.
2. **Eval sidecar:** the agent copies an adapter to a shared folder and calls one MCP tool, `evaluate_val` (30
   calls): Taboo and the 7-dataset OOD classification set. It only ever gets back aggregate numbers.
3. **Verifier (separate container, after the agent stops):** retrains the submitted code from scratch and scores it
   with its own copy of the eval code on the held-out test (Number Prediction, MMLU Prediction), which the agent never
   sees, plus validation again. It scores Taboo on a different prompt set from the one the agent's tool used.
4. **Four independent trials** give a distribution of held-out scores. Held-out scores never reach the agent.

## Appendix: results reported in the source papers

Every number comes from a printed bar label in the cited figure, or from the paper's text where noted. Values marked
≈ are read off charts that have no value labels. Scores are accuracies (0–1) unless noted. The rows closest to our
protocol are **bold**. The papers use Qwen3-8B (and other large models) with full-size training data, so these numbers
are references, not targets our 0.8B setup should match.

- **AO paper:** Karvonen et al., "Activation Oracles", arXiv 2512.15674. Code: this repo.
- **AOBench:** Bauer et al., "Building Better Activation Oracles", arXiv 2606.02609. Code: japhba/activation_oracles.

Training-mix names in the AO paper: **Full dataset** = SPQA + classification + context prediction (PastLens), 1M samples
unless noted. SPQA is LatentQA's system-prompt QA (Pan et al.). **Original model** = the untrained base model used as
the oracle.

### Taboo (AO paper)

| Model | Setting | Score | Source |
|---|---|---|---|
| Qwen3-8B | Full dataset, 1M, single token | 0.495 | Fig. 7; Fig. 12; Fig. 19 |
| Qwen3-8B | Full dataset, 400k (truncated mix) | 0.459 | Fig. 7 |
| Qwen3-8B | SPQA + classification, 400k | 0.302 | Fig. 7; Fig. 19 ("LatentQA + Classification") |
| Qwen3-8B | Classification only | 0.150 | Fig. 19 |
| Qwen3-8B | LatentQA (SPQA) only | 0.116 | Fig. 19 |
| Qwen3-8B | Original model | 0.192 | Fig. 19 |
| **Qwen3-8B** | **Full dataset, 1M, full sequence (our protocol)** | **0.285** | Fig. 12 |
| Gemma-2-9B | Full dataset, full sequence | 0.205 | Fig. 3; Figs. 12, 14, 20; text "21%" (Sec. 4.1) |
| Gemma-2-9B | Full dataset, single token | 0.312 | Fig. 12 |
| Gemma-2-9B | Full dataset, layer-75% input | 0.056 | Fig. 14 |
| Gemma-2-9B | SPQA only | 0.044 | Fig. 3; Fig. 20 |
| Gemma-2-9B | SPQA + classification | 0.052 | Fig. 20 |
| Gemma-2-9B | Classification only | 0.002 | Fig. 20 |
| Gemma-2-9B | Original model | 0.000 | Fig. 20 |
| Gemma-2-9B | Best white-box baseline (Cywiński et al.) | 0.041 | Fig. 3; text "4%" |
| Gemma-2-9B | Best black-box baseline (Cywiński et al.) | 0.072 | Fig. 3 |
| (not stated) | SPQA only, full sequence → single token | 0.044 → 0.216 | App. C.4 text |

### Taboo (AOBench)

AOBench scores are chance-adjusted, mean of 3 seeds, all runs trained on 50M tokens (Fig. 12, p. 13). The figure
doesn't name the base model; the paper's other experiments use Qwen3-8B. Recipes: *adam* = the AO-paper recipe;
*conv* = LatentQA replaced by their conversational dataset; *on-policy* = FineWeb replaced by on-policy chain-of-thought
text; *multi-layer* = multi-layer input instead of layer 18; *best v3* = multi-layer + conversational + on-policy;
*+steer 2×* = best v3 with 2× injection strength.

| adam | conv | on-policy | multi-layer | best v3 | +steer 2× | Source |
|---|---|---|---|---|---|---|
| +0.28 | +0.35 | +0.25 | +0.30 | +0.37 | +0.38 | Fig. 12 |

Also, the original AO's unfiltered single-token Taboo accuracy is 0.466. Requiring consensus ≥ 0.8 over 10 samples
keeps 19.4% of items at 94.3% precision (Sec. A.1 text; Fig. 11).

### OOD classification: the 7 held-out datasets (AO paper)

The paper reports the average over its **7 held-out (OOD) datasets** (App. C.3): the same datasets, questions and
pooled substring scoring as our OOD classification metric (experiments/plotting/plot_classification_eval.py), but with
**single-token** input, where ours reads the full sequence. It reports no per-dataset scores and no full-sequence
classification results.

| Model | Setting | OOD avg (7 datasets) | Source |
|---|---|---|---|
| **Qwen3-8B** | **Full dataset, 1M** | **0.700** | Fig. 7 |
| Qwen3-8B | Full dataset, 400k | 0.685 | Fig. 7 |
| Qwen3-8B | SPQA + classification, 400k | 0.686 | Fig. 7 |
| Qwen3-8B | SPQA only / Patchscopes | ≈0.515 / ≈0.505 | Fig. 6 |
| Llama-3.3-70B | AO / SPQA only / Patchscopes | ≈0.80 / ≈0.625 / ≈0.50 | Fig. 6 |
| Gemma-2-9B | AO / SPQA only / Patchscopes | ≈0.74 / ≈0.675 / ≈0.50 | Fig. 6 |
| Claude 3.5 Haiku | AO / SPQA only / Patchscopes | ≈0.705 / ≈0.575 / ≈0.535 | Fig. 6 |
| Llama-3.3-70B | "Classification", steering after layer 0 / layer 1 | 0.559 / 0.604 | Fig. 9 |
| Qwen3-8B | By input layer 0 / 10 / 25 / 33 / 50 / 66 / 75% | ≈0.53 / 0.62 / 0.70 / 0.72 / 0.70 / 0.68 / 0.75 | Fig. 13 |

### Classification training sets: the 7 in-distribution (IID) datasets (AO paper)

| Model | Setting | IID avg | Source |
|---|---|---|---|
| Qwen3-8B | By input layer 0 / 10 / 25 / 33 / 50 / 66 / 75% | ≈0.55 / 0.63 / 0.82 / 0.83 / 0.90 / 0.87 / 0.89 | Fig. 13 |

LatentQA and PastLens are training-only; neither paper reports a score on them.

### Number Prediction (AOBench)

Chance-adjusted (chance ≈ 0 for exact-number answers), mean of 3 seeds, 50M training tokens, last-10-token input (ours
reads the full sequence).

| **adam** | conv | on-policy | multi-layer | best v3 | +steer 2× | Source |
|---|---|---|---|---|---|---|
| **+0.00** | +0.01 | +0.01 | +0.03 | +0.15 | +0.20 | Fig. 12 |

*adam* is closest to our recipe. The AOBench paper text describes this task as predicting "the numbers involved in an
arithmetic equation"; its code (which we follow) asks for the model's own answer.

### MMLU Prediction (AOBench)

AOBench scores this task by ROC AUC on the yes/no logit margin and plots it chance-adjusted, so AUC 0.5 maps to 0
(scripts/paper_plots.py). Mean of 3 seeds, 50M training tokens. The adjustment itself happens in AOBench's external
scoring package, not in their repo, so the exact mapping back to AUC is unconfirmed: +0.28 is AUC 0.78 if the adjustment
is AUC − 0.5, or 0.64 if it is 2 × (AUC − 0.5). Our reward reports the raw mean AUC.

These are the only MMLU Prediction results in either paper: the AO paper doesn't evaluate it, and AOBench reports it
only in Fig. 12 (p. 13; individual seeds are plotted as dots without value labels).

| Model | Recipe (AOBench Fig. 12 legend) | What changes vs the AO-paper recipe | Score (chance-adjusted) | Source |
|---|---|---|---|---|
| not named (Qwen3-8B elsewhere in the paper) | **adam** | **None: the AO-paper recipe (closest to ours)** | **+0.28** | Fig. 12 |
| same | latentqa → conversational | LatentQA replaced by their conversational dataset | +0.41 | Fig. 12 |
| same | fineweb → on_policy | FineWeb PastLens text replaced by on-policy chain-of-thought text | +0.36 | Fig. 12 |
| same | L18 → multi-layer [21..25] | Multi-layer activation input instead of layer 18 | +0.39 | Fig. 12 |
| same | multi-layer + conversational + on_policy (Best v3) | All three changes above | +0.56 | Fig. 12 |
| same | + steering 2.0× | Best v3 with 2× injection strength | +0.58 | Fig. 12 |

AOBench reads the last 10 tokens. Our capped Qwen3.5-0.8B baseline scores 0.556 mean AUC on the last 10 tokens and
0.628 on the full sequence (our protocol); that is +0.128 if AOBench's adjustment is AUC − 0.5.

The AOBench paper text describes this task as predicting the answer the model would give "if it were interrupted
mid-reasoning"; its code (which we follow) uses no reasoning (thinking off) and asks whether the model is or was correct.

### Backtracking (AOBench; dropped from our scoring)

Their version is a 4-way multiple-choice question about what the model is uncertain about. Rebuilding its dataset
needs the Claude API.

| adam | conv | on-policy | multi-layer | best v3 | +steer 2× | Source |
|---|---|---|---|---|---|---|
| +0.13 | +0.05 | +0.14 | +0.13 | +0.07 | +0.13 | Fig. 12 |

The original AO scores 1.26/5 mean correctness on Qwen3-8B given only the final-token activation, and 2.10/5 given the
last 50 tokens (Sec. A.1 text; Fig. 10).
