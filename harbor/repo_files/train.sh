#!/bin/bash
# Submission entrypoint: trains the activation oracle from scratch and writes the final LoRA adapter to $OUTPUT_DIR.
set -euo pipefail
OUTPUT_DIR=${OUTPUT_DIR:-oracle_out}
# nl_probes is imported from this repo (the images install dependencies only)
export PYTHONPATH="$PWD"
START=$(date +%s)

torchrun --nproc_per_node=1 nl_probes/sft.py

# sft.py names its save dir after wandb_suffix, so take the newest final adapter and check this run wrote it
FINAL=$(ls -td checkpoints*/final | head -1)
test "$(stat -c %Y "$FINAL")" -ge "$START"
rm -rf "$OUTPUT_DIR"
cp -r "$FINAL" "$OUTPUT_DIR"
