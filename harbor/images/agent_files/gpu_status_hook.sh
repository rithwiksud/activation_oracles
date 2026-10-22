#!/bin/bash
# PostToolUse hook: appends a one-line time and GPU-use status to every tool result, so idle GPU time is visible.
status=$(/opt/ao_agent/time_left --brief)
printf '{"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": "%s"}}\n' "$status"
