#!/bin/bash
# Reference "solution" for Harbor's oracle agent: submit the unchanged baseline. Used to smoke-test the pipeline
# and to get the baseline score without spending agent time.
set -euo pipefail
test -f /app/repo/train.sh
