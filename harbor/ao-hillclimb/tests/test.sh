#!/bin/bash
# Retrains the oracle from scratch with the submitted repo, then scores it with the reference eval code (not the
# submission's): Number Prediction + MMLU Prediction (held-out test), Taboo + OOD classification (validation).
set -euo pipefail

SUBMISSION=/home/trainer/repo
OUT=/logs/verifier
TRAIN_TIMEOUT=50m

cp -r /app/repo "$SUBMISSION"
chown -R trainer:trainer "$SUBMISSION"

# runuser keeps the image environment (PATH, HF_HOME, offline flags) but drops to the unprivileged user.
# wandb runs offline here: an upload stalled at exit (table artifacts) once ran a finished retrain past the time limit.
if ! timeout "$TRAIN_TIMEOUT" runuser -u trainer -- bash -c "cd $SUBMISSION && WANDB_MODE=offline OUTPUT_DIR=$SUBMISSION/oracle_out bash train.sh" \
    > "$OUT/train.log" 2>&1; then
    echo "train.sh failed or exceeded $TRAIN_TIMEOUT, see train.log"
    echo '{"number_prediction": 0.0, "mmlu_prediction": 0.0, "taboo": 0.0, "ood_classification": 0.0}' > "$OUT/reward.json"
    exit 0
fi

# Keep the trained oracle with the trial's logs for later analysis
cp -r "$SUBMISSION/oracle_out" "$OUT/oracle_out"

export PYTHONPATH=/opt/ao_eval:/opt/ao_ref
# Smoke tests (AO_SMOKE=1, set only by the smoke-test task) score a handful of items per eval
SMALL=() TABOO_SMALL=()
if [ "${AO_SMOKE:-0}" = "1" ]; then
    SMALL=(--max-items 16) TABOO_SMALL=(--max-words 2 --max-prompts 3)
fi
python /opt/ao_eval/eval_classification.py --oracle-lora "$SUBMISSION/oracle_out" --data-dir /opt/ao_eval/classification_test \
    "${SMALL[@]}" --output "$OUT/ood_classification.json" > "$OUT/ood_classification.log" 2>&1
# Taboo re-measured on the paper's direct test prompts; the agent's eval tool uses the standard val prompts
python /opt/ao_eval/eval_taboo.py --oracle-lora "$SUBMISSION/oracle_out" \
    --context-prompts /opt/ao_heldout/taboo_direct_test.txt "${TABOO_SMALL[@]}" --output "$OUT/taboo.json" > "$OUT/taboo.log" 2>&1
# Held-out test tasks
python /opt/ao_heldout/eval_number_prediction.py --oracle-lora "$SUBMISSION/oracle_out" "${SMALL[@]}" \
    --output "$OUT/number_prediction.json" > "$OUT/number_prediction.log" 2>&1
python /opt/ao_heldout/eval_mmlu_prediction.py --oracle-lora "$SUBMISSION/oracle_out" "${SMALL[@]}" \
    --output "$OUT/mmlu_prediction.json" > "$OUT/mmlu_prediction.log" 2>&1

# Held-out scores are deliberately not logged to wandb: agents in later trials have a key that can read the project.
python - <<EOF
import json
summary = lambda name: json.load(open(f"$OUT/{name}.json"))["summary"]["accuracy"]
reward = {
    "number_prediction": summary("number_prediction"),  # held-out test
    "mmlu_prediction": summary("mmlu_prediction"),  # held-out test (mean ROC AUC, chance 0.5)
    "taboo": summary("taboo"),  # validation, scored on the paper's direct test prompts (eval tool: standard val prompts)
    "ood_classification": summary("ood_classification"),  # validation (the AO paper's 7 OOD classification datasets)
}
json.dump(reward, open("$OUT/reward.json", "w"))
print(reward)
EOF
