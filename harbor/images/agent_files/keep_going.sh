#!/bin/bash
# Stop hook: a headless session tends to declare itself done early. Block stopping until the last 15 minutes of the
# budget, so the agent keeps researching. Budget must match [agent].timeout_sec in task.toml.
BUDGET_MIN=150
MIN_STOP_MIN=135
used=$(( ($(date +%s) - $(cat /app/.agent_start)) / 60 ))
if [ "$used" -lt "$MIN_STOP_MIN" ]; then
    reason="Only $used of $BUDGET_MIN minutes used, so don't stop yet. Keep researching: pick the next hypothesis, test it with a proxy run, confirm promising changes with a full run, commit improvements and update NOTES.md. Stopping is allowed after $MIN_STOP_MIN minutes."
    printf '{"decision": "block", "reason": "%s"}\n' "$reason"
fi
