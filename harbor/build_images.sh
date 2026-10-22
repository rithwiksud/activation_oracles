#!/bin/bash
# Stages build contexts and builds the local images used by harbor/ao-hillclimb:
#   ao-base        venv, Qwen3.5-0.8B, offline training data, reference repo (no eval code, targets or eval data)
#   ao-agent       ao-base + git working copy at /app/repo + Claude Code
#   ao-evalserver  ao-base + validation evals (Taboo, the AO paper's 7 OOD classification sets) behind a capped MCP tool
#   ao-verifier    ao-base + the same evals + held-out Number Prediction / MMLU Prediction + tests/test.sh (root-only)
#
# Prerequisites (run once, from the repo root):
#   uv sync --extra linear-attn                                   # host venv with compiled flash-attn/causal-conv1d
#   python harbor/scripts/dump_pastlens_slices.py $BUILD/ao_data_full 60000      # full-size; capped per build below
#   python harbor/heldout/make_number_prediction.py $BUILD/heldout/number_prediction_aobench.json \
#       <AOBench clone>/data_pipelines/number_prediction/Qwen3-8B/number_prediction_eval_dataset.json
#   python harbor/heldout/make_mmlu_prediction.py $BUILD/heldout/mmlu_prediction.json
#   python harbor/scripts/train_targets.py $BUILD/targets taboo <unused personaqa dir>   # Taboo targets, trained once
#   python harbor/scripts/build_classification_test_sets.py $BUILD/classification_test_fullseq   # full-sequence input
set -euo pipefail
# The staging scripts import the staged code; keep its bytecode out of the agent's committed baseline repo
export PYTHONDONTWRITEBYTECODE=1

REPO=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${BUILD:-/home/ubuntu/ao_harbor_build}
SITE=$REPO/.venv/lib/python3.10/site-packages
BASE_CTX=$BUILD/base
EVAL_CTX=$BUILD/evalserver
VERIFIER_CTX=$BUILD/verifier

# ---- ao-base context ----
rm -rf "$BASE_CTX/prebuilt" "$BASE_CTX/ao_ref" "$BASE_CTX/ao_eval" "$BASE_CTX/ao_targets"
mkdir -p "$BASE_CTX/prebuilt" "$BASE_CTX/ao_ref/datasets"
rm -rf "$BASE_CTX/ao_data"
cp -r "$BUILD/ao_data_full" "$BASE_CTX/ao_data"

cp "$REPO/pyproject.toml" "$REPO/uv.lock" "$BASE_CTX/"
for f in flash_attn flash_attn-2.8.3.dist-info flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so hopper \
    causal_conv1d causal_conv1d-1.7.0.dist-info causal_conv1d_cuda.cpython-310-x86_64-linux-gnu.so; do
    cp -r "$SITE/$f" "$BASE_CTX/prebuilt/"
done

# Reference repo, built from an allowlist, then stripped of check-task eval data
rsync -a --exclude __pycache__ --exclude trl_training/taboo_train.py "$REPO/nl_probes" "$BASE_CTX/ao_ref/"
for d in classification_datasets latentqa_datasets; do
    rsync -a "$REPO/datasets/$d" "$BASE_CTX/ao_ref/datasets/"
done
cp "$REPO/pyproject.toml" "$REPO/harbor/repo_files/train.sh" "$BASE_CTX/ao_ref/"
# PastLens streams from the HF hub by default; point it at the local slices
sed -i -e 's#"HuggingFaceFW/fineweb"#"/opt/ao_data/fineweb"#' -e 's#"lmsys/lmsys-chat-1m"#"/opt/ao_data/lmsys-chat-1m"#' \
    "$BASE_CTX/ao_ref/nl_probes/dataset_classes/past_lens_dataset.py"
grep -q '"/opt/ao_data/fineweb"' "$BASE_CTX/ao_ref/nl_probes/dataset_classes/past_lens_dataset.py"
"$REPO/.venv/bin/python" "$REPO/harbor/scripts/strip_eval_data.py" "$BASE_CTX/ao_ref"
# Training data is capped at the baseline's 10x-reduced mixture; nothing more ships in the images
"$REPO/.venv/bin/python" "$REPO/harbor/scripts/cap_training_data.py" "$BASE_CTX/ao_ref" "$BASE_CTX/ao_data"
if find "$BASE_CTX/ao_ref" -name __pycache__ | grep -q .; then
    echo "__pycache__ found in the agent-visible tree" && exit 1
fi
# Eval identifiers must not appear anywhere in the agent-visible tree
EVAL_IDS="number_prediction|mmlu_prediction|eval_backtracking|ao_heldout|ao_eval|ao_targets|taboo_direct|eval_server"
if grep -rniE "$EVAL_IDS" "$BASE_CTX/ao_ref"; then
    echo "Eval identifiers found in the agent-visible tree" && exit 1
fi

docker build -t ao-base:latest -f "$REPO/harbor/images/base.Dockerfile" "$BASE_CTX"
docker build -t ao-agent:latest -f "$REPO/harbor/images/agent.Dockerfile" "$REPO/harbor/images"

# ---- Validation eval assets, shared by the eval server and the verifier ----
stage_evals() {
    mkdir -p "$1/ao_eval/classification_test" "$1/ao_targets"
    cp "$REPO/harbor/evals/ao_eval_common.py" "$REPO/harbor/evals/eval_taboo.py" \
        "$REPO/harbor/evals/eval_classification.py" "$REPO/datasets/taboo/taboo_standard_val.txt" \
        "$1/ao_eval/"
    for ds in ag_news language_identification singular_plural engels_headline_istrump engels_headline_isobama \
        engels_headline_ischina engels_hist_fig_ismale; do
        test "$(ls "$BUILD"/classification_test_fullseq/classification_${ds}_*_test_*.pt | wc -l)" = 1
        cp "$BUILD"/classification_test_fullseq/classification_${ds}_*_test_*.pt "$1/ao_eval/classification_test/"
    done
    cp -r "$BUILD/targets/taboo" "$1/ao_targets/"
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
    "$REPO/harbor/evals/eval_mmlu_prediction.py" "$BUILD/heldout/mmlu_prediction.json" "$VERIFIER_CTX/ao_heldout/"
cp "$BUILD/heldout/number_prediction_aobench.json" "$VERIFIER_CTX/ao_heldout/number_prediction.json"
cp -r "$REPO/harbor/ao-hillclimb/tests" "$VERIFIER_CTX/tests"
docker build -t ao-verifier:latest -f "$REPO/harbor/images/verifier.Dockerfile" "$VERIFIER_CTX"
