#!/bin/bash
# Stages build contexts and builds the local images used by harbor/ao-hillclimb:
#   ao-base        venv, Qwen3.5-0.8B, offline training data, reference repo (no eval code, targets or eval data)
#   ao-agent       ao-base + git working copy at /app/repo + Claude Code
#   ao-evalserver  ao-base + train/check-task evals (Taboo, PersonaQA, OOD classification) behind a capped MCP tool
#   ao-verifier    ao-base + the same evals + held-out Number Prediction / Backtracking + tests/test.sh (root-only)
#
# Prerequisites (run once, from the repo root):
#   uv sync --extra linear-attn                                   # host venv with compiled flash-attn/causal-conv1d
#   python harbor/scripts/dump_pastlens_slices.py $BUILD/base/ao_data 60000
#   python harbor/heldout/make_number_prediction.py $BUILD/heldout/number_prediction.json 400 20261021
#   python harbor/heldout/make_backtracking.py $BUILD/heldout/backtracking.json test 1319 400 20261021
#   unzip datasets/personaqa_data.zip -d $BUILD && python harbor/scripts/train_targets.py $BUILD/targets all $BUILD/personaqa_data/shuffled
#   torchrun nl_probes/sft.py once, so sft_training_data/ holds the cached OOD classification test sets
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${BUILD:-/home/ubuntu/ao_harbor_build}
SITE=$REPO/.venv/lib/python3.10/site-packages
BASE_CTX=$BUILD/base
EVAL_CTX=$BUILD/evalserver
VERIFIER_CTX=$BUILD/verifier

# ---- ao-base context ----
rm -rf "$BASE_CTX/prebuilt" "$BASE_CTX/ao_ref" "$BASE_CTX/ao_eval" "$BASE_CTX/ao_targets"
mkdir -p "$BASE_CTX/prebuilt" "$BASE_CTX/ao_ref/datasets"
test -f "$BASE_CTX/ao_data/fineweb/train.jsonl"
test -f "$BASE_CTX/ao_data/lmsys-chat-1m/train.jsonl"

cp "$REPO/pyproject.toml" "$REPO/uv.lock" "$BASE_CTX/"
for f in flash_attn flash_attn-2.8.3.dist-info flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so hopper \
    causal_conv1d causal_conv1d-1.7.0.dist-info causal_conv1d_cuda.cpython-310-x86_64-linux-gnu.so; do
    cp -r "$SITE/$f" "$BASE_CTX/prebuilt/"
done

# Reference repo, built from an allowlist, then stripped of check-task eval data
rsync -a --exclude __pycache__ --exclude trl_training/taboo_train.py "$REPO/nl_probes" "$BASE_CTX/ao_ref/"
for d in classification_datasets latentqa_datasets personaqa_data; do
    rsync -a "$REPO/datasets/$d" "$BASE_CTX/ao_ref/datasets/"
done
cp "$REPO/pyproject.toml" "$REPO/harbor/repo_files/train.sh" "$BASE_CTX/ao_ref/"
# PastLens streams from the HF hub by default; point it at the local slices
sed -i -e 's#"HuggingFaceFW/fineweb"#"/opt/ao_data/fineweb"#' -e 's#"lmsys/lmsys-chat-1m"#"/opt/ao_data/lmsys-chat-1m"#' \
    "$BASE_CTX/ao_ref/nl_probes/dataset_classes/past_lens_dataset.py"
grep -q '"/opt/ao_data/fineweb"' "$BASE_CTX/ao_ref/nl_probes/dataset_classes/past_lens_dataset.py"
"$REPO/.venv/bin/python" "$REPO/harbor/scripts/strip_eval_data.py" "$BASE_CTX/ao_ref"
# Eval identifiers must not appear anywhere in the agent-visible tree
EVAL_IDS="number_prediction|eval_backtracking|ao_heldout|ao_eval|ao_targets|taboo_direct|eval_server"
if grep -rniE "$EVAL_IDS" "$BASE_CTX/ao_ref"; then
    echo "Eval identifiers found in the agent-visible tree" && exit 1
fi

docker build -t ao-base:latest -f "$REPO/harbor/images/base.Dockerfile" "$BASE_CTX"
docker build -t ao-agent:latest -f "$REPO/harbor/images/agent.Dockerfile" "$REPO/harbor/images"

# ---- Train/check-task eval assets, shared by the eval server and the verifier ----
stage_evals() {
    mkdir -p "$1/ao_eval/classification_test" "$1/ao_targets"
    cp "$REPO/harbor/evals/ao_eval_common.py" "$REPO/harbor/evals/eval_personaqa.py" "$REPO/harbor/evals/eval_taboo.py" \
        "$REPO/harbor/evals/eval_classification.py" "$REPO/datasets/taboo/taboo_standard_val.txt" \
        "$REPO/datasets/personaqa_data/shuffled/personas.jsonl" "$1/ao_eval/"
    for ds in ag_news language_identification singular_plural; do
        ls "$REPO"/sft_training_data/classification_${ds}_model_Qwen3_5-0_8B_n_250_*_test_*.pt > /dev/null
        cp "$REPO"/sft_training_data/classification_${ds}_model_Qwen3_5-0_8B_n_250_*_test_*.pt "$1/ao_eval/classification_test/"
    done
    cp -r "$BUILD/targets/personaqa" "$BUILD/targets/taboo" "$1/ao_targets/"
}

# ---- ao-evalserver context ----
rm -rf "$EVAL_CTX"
stage_evals "$EVAL_CTX"
cp -r "$REPO/harbor/eval_server" "$EVAL_CTX/eval_server"
docker build -t ao-evalserver:latest -f "$REPO/harbor/images/evalserver.Dockerfile" "$EVAL_CTX"

# ---- ao-verifier context ----
rm -rf "$VERIFIER_CTX"
stage_evals "$VERIFIER_CTX"
mkdir -p "$VERIFIER_CTX/ao_heldout"
cp "$REPO/datasets/taboo/taboo_direct_test.txt" "$REPO/harbor/evals/eval_number_prediction.py" \
    "$REPO/harbor/evals/eval_backtracking.py" "$BUILD/heldout/number_prediction.json" "$BUILD/heldout/backtracking.json" \
    "$VERIFIER_CTX/ao_heldout/"
cp -r "$REPO/harbor/ao-hillclimb/tests" "$VERIFIER_CTX/tests"
docker build -t ao-verifier:latest -f "$REPO/harbor/images/verifier.Dockerfile" "$VERIFIER_CTX"
