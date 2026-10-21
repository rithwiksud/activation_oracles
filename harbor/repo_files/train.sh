#!/bin/bash
# Submission entrypoint: trains the activation oracle from scratch and writes the final LoRA adapter to $OUTPUT_DIR.
set -euo pipefail
OUTPUT_DIR=${OUTPUT_DIR:-oracle_out}
# nl_probes is imported from this repo (the images install dependencies only)
export PYTHONPATH="$PWD"

torchrun --nproc_per_node=1 nl_probes/sft.py

rm -rf "$OUTPUT_DIR"
cp -r checkpoints_latentqa_cls_past_lens_Qwen3_5-0_8B/final "$OUTPUT_DIR"
