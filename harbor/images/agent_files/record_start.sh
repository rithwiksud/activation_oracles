#!/bin/bash
# SessionStart hook: records when the agent first started and starts the GPU sampler. Later SessionStart events
# (resume, compaction) keep both.
[ -f /app/.agent_start ] || date +%s > /app/.agent_start
if [ ! -f /var/tmp/ao_gpu_sampler.pid ] || ! kill -0 "$(cat /var/tmp/ao_gpu_sampler.pid)" 2>/dev/null; then
    setsid nohup /opt/ao_agent/gpu_sampler.sh > /dev/null 2>&1 < /dev/null &
    echo $! > /var/tmp/ao_gpu_sampler.pid
fi
