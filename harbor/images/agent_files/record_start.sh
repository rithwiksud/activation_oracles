#!/bin/bash
# SessionStart hook: records when the agent first started. Later SessionStart events (resume, compaction) keep it.
[ -f /app/.agent_start ] || date +%s > /app/.agent_start
