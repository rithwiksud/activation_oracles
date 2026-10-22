# Harbor: autoresearch hill climbing for activation oracle training

Claude Code (authenticated with a Claude subscription) works in a GPU sandbox for 2 hours per trial, improving
activation oracle training for Qwen3.5-0.8B. Its submission is then retrained from scratch in a separate verifier
sandbox and scored.

| Tier | Tasks | Agent sees it? |
| - | - | - |
| Validation | Taboo (20 secret-word targets, the paper's standard val prompts), OOD classification (the AO paper's 7 held-out classification datasets, pooled) | Score only, via `evaluate_val` (30 calls) |
| Test (held out) | Number Prediction and MMLU Prediction (AOBench's protocols) on the base model | No |

Training data (LatentQA, classification, PastLens) is fully visible to the agent and editable. Validation evals
run in an `eval-server` sidecar that the agent reaches only through MCP; their code, targets and data are not in its sandbox.

`reward.json` reports `number_prediction` and `mmlu_prediction` (the held-out scores), plus `taboo` (re-measured on the
paper's direct test prompts) and `ood_classification`. Held-out scores never go to wandb, because the agent's key can read the entity.

## Layout

| Path | What |
| - | - |
| `ao-hillclimb/` | The Harbor task (`task.toml`, `instruction.md`, `tests/test.sh`, baseline `solution/`) |
| `images/` | Dockerfiles for `ao-base`, `ao-agent`, `ao-verifier` |
| `scripts/build_classification_test_sets.py` | Builds the classification test sets with the AO paper's single-token protocol |
| `build_images.sh` | Stages contexts under `$BUILD` (default `/home/ubuntu/ao_harbor_build`) and builds the images |
| `evals/` | Eval scripts, each reading the positions its source paper reports (Taboo: single token, classification: single token, Number Prediction and MMLU Prediction: last 10 tokens) |
| `scripts/train_targets.py` | Trains the 20 Taboo target LoRAs for Qwen3.5-0.8B, once each (PersonaQA support is unused) |
| `scripts/dump_pastlens_slices.py` | Dumps fineweb / lmsys-chat slices so PastLens works offline |
| `repo_files/train.sh` | The submission entrypoint placed in the agent's repo |

## Isolation

- The agent's repo is built from an allowlist (`nl_probes/` minus `taboo_train.py`, and the classification and LatentQA
  datasets), with no git history, and the build fails if held-out task identifiers appear in it.
- The agent can reach only `api.anthropic.com` and `api.wandb.ai` while it runs; the verifier only `api.wandb.ai`. Models and data are preloaded.
- The verifier image holds the held-out data, evals, Taboo test prompts and `tests/` with root-only permissions. The submitted `train.sh` runs as
  an unprivileged `trainer` user, so it can't read them.
- Evals run with the reference copy of the code (`/opt/ao_ref`), not the agent's, so the agent can't change scoring.

## One-time setup

```bash
uv sync --extra linear-attn
python harbor/scripts/dump_pastlens_slices.py /home/ubuntu/ao_harbor_build/base/ao_data 60000
unzip datasets/personaqa_data.zip -d /home/ubuntu/ao_harbor_build   # train_targets.py takes the path even for Taboo only
python harbor/scripts/train_targets.py /home/ubuntu/ao_harbor_build/targets taboo /home/ubuntu/ao_harbor_build/personaqa_data/shuffled   # trained once
bash harbor/build_images.sh
claude setup-token   # prints a CLAUDE_CODE_OAUTH_TOKEN for your subscription
```

## Running

```bash
# Smoke test / baseline score: submits the unchanged baseline, no Claude usage
harbor run -p harbor/ao-hillclimb -a oracle -e docker -n 1 -o /home/ubuntu/ao_harbor_build/jobs

# Real run (harbor.env holds CLAUDE_CODE_OAUTH_TOKEN, WANDB_API_KEY and WANDB_ENTITY; mode 600, outside the repo)
harbor run -p harbor/ao-hillclimb -a claude-code -m anthropic/claude-opus-5-5 -e docker \
  --env-file /home/ubuntu/ao_harbor_build/harbor.env -y -k 4 -n 1 -o /home/ubuntu/ao_harbor_build/jobs
```

There's one GPU, so `-n 1` is required (Harbor defaults to 4 concurrent trials), and don't run other GPU jobs
during a trial. Results go to `/home/ubuntu/ao_harbor_build/jobs`; browse them with `harbor view`.
